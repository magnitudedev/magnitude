//! Exact import-time operations on stored tensors (model-family plan §2.3).
//!
//! A family names them in `WeightDescriptor::transforms`. They run on the
//! host over the stored bytes before the ordinary dense or repack import, so
//! the device importers only ever see a plain tensor of the logical shape.
//! Row operations move whole encoded rows and are exact for every encoding;
//! a scale multiplies F32 values once in F32. A second-level scale
//! (`ScaleByTensor`) leaves the stored bytes as they are: it becomes resident
//! beside the weight (`WeightPlan::scale`) and entries apply it to their F32
//! accumulator. A grid of block scales (`ScaleBlocks`) is the source's
//! second plane: every later transform moves its rows with the rows of
//! codes they scale, and the source's bytes are planar as the file stores
//! them, each matrix's codes then its grid (`seismic::GgufScales::Grid`).

use crate::residency::Stored;
use magnitude_family_contracts::{
    ImportTransform, ProgressiveCodeDepth, ProgressivePlane, ScaleCodes, ScaledActivations,
    WeightDescriptor,
};
use seismic::{Element, GgufScales, Grid, GridScale};

/// Whether the importer can apply `descriptor`'s transforms to a tensor
/// stored as `source` with `stored` shape, making the shape `descriptor`
/// declares: [`admit_transforms`], checked against it.
pub(crate) fn admit(
    descriptor: &WeightDescriptor,
    stored: &[u64],
    source: Element,
) -> Result<Vec<u64>, String> {
    let shape = admit_transforms(descriptor, stored, source)?;
    if shape != descriptor.shape {
        return Err(format!(
            "{:?}: stored shape {stored:?} transforms to {shape:?}, not {:?}",
            descriptor.name, descriptor.shape
        ));
    }
    Ok(shape)
}

/// Whether the importer can apply `descriptor`'s transforms to a tensor
/// stored as `source` with `stored` shape (whatever shape `descriptor`
/// declares), and the logical shape they produce. Checked at plan time, so
/// an unsupported transform makes the model's plan fail rather than its
/// load.
pub(crate) fn admit_transforms(
    descriptor: &WeightDescriptor,
    stored: &[u64],
    source: Element,
) -> Result<Vec<u64>, String> {
    let dense = source.dtype().is_some();
    // Under a grid a row of its scales scales this many rows of codes.
    let tile_rows = grid(descriptor).map_or(1, |(_, grid)| grid.tile[0]);
    let mut shape = stored.to_vec();
    for (index, transform) in descriptor.transforms.iter().enumerate() {
        match transform {
            ImportTransform::ScaleBlocks { tensor, .. } if index != 0 => {
                return Err(format!(
                    "{:?}: scale grid {tensor:?} scales the tensor as stored, so it is its \
                     first transform",
                    descriptor.name
                ));
            }
            // The grid's rows move with the rows of codes they scale.
            ImportTransform::Rows(range)
                if !(range.start.is_multiple_of(tile_rows) && range.rows.is_multiple_of(tile_rows)) =>
            {
                return Err(format!(
                    "{:?}: rows {}..{} do not start and end on its scale grid's tiles of \
                     {tile_rows} rows",
                    descriptor.name,
                    range.start,
                    range.start + range.rows
                ));
            }
            ImportTransform::PermuteRows { .. } if tile_rows != 1 => {
                return Err(format!(
                    "{:?}: its rows share scales in tiles of {tile_rows} rows, so they cannot be \
                     permuted",
                    descriptor.name
                ));
            }
            ImportTransform::Rows(_) | ImportTransform::PermuteRows { .. }
                if shape.len() == 1 && !dense =>
            {
                return Err(format!(
                    "{:?}: element rows of a packed vector cannot be selected",
                    descriptor.name
                ));
            }
            ImportTransform::Scale { .. } if source != Element::f32() => {
                return Err(format!(
                    "{:?}: a scale is exact only on F32 storage, not {}",
                    descriptor.name,
                    source.name()
                ));
            }
            ImportTransform::ScaleByTensor { tensor, .. }
            | ImportTransform::ScaleBlocks { tensor, .. }
                if descriptor
                    .transforms
                    .iter()
                    .filter(|other| {
                        matches!(
                            other,
                            ImportTransform::Flatten
                                | ImportTransform::ScaleByTensor { .. }
                                | ImportTransform::ScaleBlocks { .. }
                        )
                    })
                    .count()
                    > 1 =>
            {
                return Err(format!(
                    "{:?}: companion scale tensor {tensor:?} must be the weight's only scale \
                     of its matrices",
                    descriptor.name
                ));
            }
            ImportTransform::Progressive { .. } => {
                if placement(descriptor, stored, source)?.is_none() {
                    return Err(format!(
                        "{:?}: a progressive plane must be its descriptor's only transform",
                        descriptor.name
                    ));
                }
            }
            _ => {}
        }
        shape = WeightDescriptor {
            transforms: vec![transform.clone()],
            ..descriptor.clone()
        }
        .transformed_shape(&shape)
        .map_err(|error| error.to_string())?;
    }
    Ok(shape)
}

/// The progressive placement at `depth` of a tensor stored as `source` with
/// `stored` shape: the one admission of a progressive matrix. `None` unless
/// it is a matrix whose columns are nonzero and fit `u32`, and whose rows
/// are whole packets of codes of `depth` that every plane's placement
/// places ([`Placement::of`](crate::progressive::Placement::of)).
pub(crate) fn progressive_matrix(
    depth: ProgressiveCodeDepth,
    stored: &[u64],
    source: Element,
) -> Option<crate::progressive::Placement> {
    let [_, columns] = stored else {
        return None;
    };
    let columns = std::num::NonZeroU32::new(u32::try_from(*columns).ok()?)?;
    crate::progressive::Placement::of(depth, source, columns)
}

/// The progressive plane `descriptor` places
/// ([`WeightDescriptor::progressive`]) of a matrix stored as `source` with
/// `stored` shape, and its placement, if it places one, as
/// [`admit_transforms`] admits it at plan time.
pub(crate) fn placement(
    descriptor: &WeightDescriptor,
    stored: &[u64],
    source: Element,
) -> Result<Option<(ProgressivePlane, crate::progressive::Placement)>, String> {
    let Some((plane, depth)) = descriptor.progressive() else {
        return Ok(None);
    };
    let placement = progressive_matrix(depth, stored, source).ok_or_else(|| {
        format!(
            "{:?}: a progressive plane must place a matrix (columns nonzero and within \
             u32) of whole packets of codes of its depth, whose placed rows' bit \
             positions are within u32",
            descriptor.name
        )
    })?;
    Ok(Some((plane, placement)))
}

/// The logical source bytes of a tensor with `stored` shape, after
/// `descriptor`'s transforms (admitted by [`admit`]); `grid` is the stored
/// bytes of its grid of block scales, where it has one, and the result then
/// planar: each matrix's codes, then its grid ([`Planes::planar`]).
pub(crate) fn apply(
    descriptor: &WeightDescriptor,
    stored: &[u64],
    bytes: Vec<u8>,
    grid: Option<&[u8]>,
) -> Result<Vec<u8>, String> {
    let mut shape = stored.to_vec();
    let mut planes = Planes { own: bytes, grid: None };
    for transform in &descriptor.transforms {
        match transform {
            ImportTransform::ScaleBlocks { tile, codes, .. } => {
                let grid = grid.expect("a weight under a scale grid is stored with it");
                if *codes == ScaleCodes::F32 {
                    for (at, value) in grid.chunks_exact(4).enumerate() {
                        f32_scale(f32::from_le_bytes(value.try_into().expect("four bytes")))
                            .map_err(|reason| format!("scale {at} of the grid: {reason}"))?;
                    }
                }
                planes.grid = Some((grid.to_vec(), to_usize(tile[0])?));
            }
            ImportTransform::Rows(range) => {
                let rows = RowLayout::of(&shape)?;
                let (start, count) = (to_usize(range.start)?, to_usize(range.rows)?);
                for (plane, per) in planes.each() {
                    *plane = rows.select(plane, per, start, count);
                }
                *shape
                    .iter_mut()
                    .rev()
                    .nth(rows.axis_from_end)
                    .expect("the row axis exists") = range.rows;
            }
            ImportTransform::PermuteRows { order } => {
                let rows = RowLayout::of(&shape)?;
                let order = order.iter().map(|&row| to_usize(row)).collect::<Result<Vec<_>, _>>()?;
                for (plane, per) in planes.each() {
                    assert_eq!(per, 1, "an admitted permutation moves rows of their own scales");
                    *plane = rows.permute(plane, &order);
                }
            }
            ImportTransform::Flatten => shape = vec![shape.iter().product()],
            ImportTransform::DropUnitAxis => {
                shape.remove(0);
            }
            ImportTransform::Scale { factor } => {
                let factor = *factor as f32;
                for word in planes.own.chunks_exact_mut(4) {
                    let value = f32::from_le_bytes(word.try_into().expect("four bytes"));
                    word.copy_from_slice(&(value * factor).to_le_bytes());
                }
            }
            // Resident beside the weight, not applied to its bytes.
            ImportTransform::ScaleByTensor { .. } => {}
            ImportTransform::Progressive { .. } => {
                return Err(format!(
                    "{:?}: a progressive plane is placed by `upload_bytes`",
                    descriptor.name
                ));
            }
        }
    }
    Ok(planes.planar(&shape))
}

/// A tensor's bytes as [`apply`] moves them: its own, and under a grid of
/// block scales the grid's, each of whose rows scales a tile's rows of the
/// tensor (the count beside it).
struct Planes {
    own: Vec<u8>,
    grid: Option<(Vec<u8>, usize)>,
}

impl Planes {
    /// Each plane, with the rows of the tensor each of its rows covers.
    fn each(&mut self) -> impl Iterator<Item = (&mut Vec<u8>, usize)> {
        std::iter::once((&mut self.own, 1))
            .chain(self.grid.as_mut().map(|(grid, rows)| (grid, *rows)))
    }

    /// The bytes of the tensor of `shape`: its own, or under a grid planar,
    /// as the source lays them (`seismic::GgufScales::Grid`): each matrix's
    /// (every axis before the last two counts matrices) codes, then its
    /// grid.
    fn planar(self, shape: &[u64]) -> Vec<u8> {
        let Some((grid, _)) = self.grid else {
            return self.own;
        };
        if self.own.is_empty() && grid.is_empty() {
            return Vec::new();
        }
        let matrices = shape[..shape.len().saturating_sub(2)].iter().product::<u64>() as usize;
        let (codes, scales) = (self.own.len() / matrices, grid.len() / matrices);
        assert!(
            self.own.len() == codes * matrices && grid.len() == scales * matrices,
            "a stack's codes and grid are its matrices'"
        );
        self.own
            .chunks_exact(codes)
            .zip(grid.chunks_exact(scales))
            .flat_map(|(codes, grid)| [codes, grid].concat())
            .collect()
    }
}

/// The bytes the importer uploads for `plan`'s weight, `bytes` the stored
/// range of `stored`: the transformed bytes in the plan's upload
/// representation. A packed weight whose upload differs from its source is
/// dequantized exactly to F32; a progressive plane is placed from the codes
/// of the stored rows ([`placement`]).
pub(crate) fn upload_bytes(
    plan: &crate::WeightPlan,
    stored: &Stored,
    bytes: Vec<u8>,
) -> Result<Vec<u8>, String> {
    let shape = stored.shape();
    if let Some((plane, placement)) = placement(&plan.descriptor, shape, plan.source)? {
        return Ok(crate::progressive::plane(plane, &placement, &bytes));
    }
    let grid = stored
        .grid()
        .map(|grid| grid.read().map_err(|error| error.to_string()))
        .transpose()?;
    let (descriptor, source, upload) = (&plan.descriptor, plan.source, plan.upload);
    let transformed = apply(descriptor, shape, bytes, grid.as_deref())?;
    if upload == source {
        return Ok(transformed);
    }
    dequantize(source, &descriptor.shape, &transformed)
}

/// Where the scales of a weight `descriptor` names are
/// (`seismic::GgufScales`): in its stored tensor's own blocks, or under its
/// grid of block scales.
pub(crate) fn stored_scales(descriptor: &WeightDescriptor) -> GgufScales {
    grid(descriptor).map_or(GgufScales::Own, |(_, grid)| GgufScales::Grid(grid))
}

/// The grid of block scales of `codes` tiling a matrix by `tile`: the one
/// map from a family's grid (`ImportTransform::ScaleBlocks`) to the
/// registry's (`seismic::Grid`), whose source a weight under it is read as
/// when the registry registers one.
pub(crate) fn scale_grid(codes: ScaleCodes, tile: [u64; 2]) -> Grid {
    let scale = match codes {
        ScaleCodes::Ue8m0 => GridScale::E8m0,
        ScaleCodes::F32 => GridScale::F32,
    };
    Grid { scale, tile }
}

/// The stored grid of block scales `descriptor` names
/// (`ImportTransform::ScaleBlocks`), with what it is ([`scale_grid`]), when
/// it has one (at most one, [`admit`]).
pub(crate) fn grid(descriptor: &WeightDescriptor) -> Option<(&str, Grid)> {
    descriptor
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ImportTransform::ScaleBlocks {
                tensor,
                tile,
                codes,
            } => Some((tensor.as_str(), scale_grid(*codes, *tile))),
            _ => None,
        })
}

/// The stored tensors one row of `descriptor`'s matrix is read from, in the
/// order [`assemble_row`] takes their ranges: its own tensor, then its grid
/// of block scales where it has one (the importer's two inputs, `Stored`).
pub(crate) fn row_tensors(descriptor: &WeightDescriptor) -> Vec<&str> {
    std::iter::once(descriptor.name.as_str())
        .chain(grid(descriptor).map(|(name, _)| name))
        .collect()
}

/// One row of `width` values of `descriptor`'s matrix as its source holds
/// it, from `ranges`, that row's range of each of its [`row_tensors`] in
/// order: the importer's own transforms ([`apply`]) over a `[1, width]`
/// matrix, under a grid planar (the row's codes, then its row of the grid).
/// The plan admitted only transforms that act within a row, so a row's
/// ranges are its whole input.
pub(crate) fn assemble_row(
    descriptor: &WeightDescriptor,
    width: u64,
    ranges: &[&[u8]],
) -> Result<Vec<u8>, String> {
    let [own, grid @ ..] = ranges else {
        return Err(format!("{:?}: a row is read from no range", descriptor.name));
    };
    if ranges.len() != row_tensors(descriptor).len() {
        return Err(format!(
            "{:?}: {} ranges of a row read from {} tensors",
            descriptor.name,
            ranges.len(),
            row_tensors(descriptor).len()
        ));
    }
    apply(descriptor, &[1, width], own.to_vec(), grid.first().copied())
}

/// Whether `scale`, an F32 block scale (`ScaleCodes::F32`), is one the
/// engine reads: finite, and zero or of magnitude at least 2^-117. A
/// value is the code times its scale, one F32 product (`fp8g128`), which
/// every decoder forms exactly; the bound keeps every nonzero product, at
/// least `2^-117 · 2^-9` (E4M3's least value), an F32 normal, which a
/// device that flushes subnormals keeps too. A checkpoint's scales are
/// amax / 448, far above it.
fn f32_scale(scale: f32) -> Result<(), String> {
    // 2^-117: biased exponent 127 - 117, no mantissa.
    const LEAST: f32 = f32::from_bits((127 - 117) << 23);
    if !scale.is_finite() {
        return Err(format!("{scale} is not finite"));
    }
    if scale != 0.0 && scale.abs() < LEAST {
        return Err(format!(
            "{scale:e} is below 2^-117, where a code's value under it can be an F32 subnormal"
        ));
    }
    Ok(())
}

/// The stored tensor holding `descriptor`'s second-level scale, when it has
/// one (at most one, [`admit`]).
pub(crate) fn scale_tensor(descriptor: &WeightDescriptor) -> Option<&str> {
    descriptor
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ImportTransform::ScaleByTensor { tensor, .. } => Some(tensor.as_str()),
            _ => None,
        })
}

/// The activations `descriptor`'s products take: its second-level scale's,
/// the rounded activation without one.
pub(crate) fn scaled_activations(descriptor: &WeightDescriptor) -> ScaledActivations {
    descriptor
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ImportTransform::ScaleByTensor { activations, .. } => Some(*activations),
            _ => None,
        })
        .unwrap_or(ScaledActivations::Rounded)
}

/// How a tensor's rows divide into matrices, which its row transforms
/// address.
struct RowLayout {
    /// Matrices of a stack (1 for a matrix or vector).
    matrices: usize,
    /// Rows of each matrix.
    rows: usize,
    /// The row axis counted from the last axis.
    axis_from_end: usize,
}

impl RowLayout {
    fn of(shape: &[u64]) -> Result<Self, String> {
        let (matrices, rows, axis_from_end) = match shape {
            [elements] => (1, *elements, 0),
            [leading @ .., rows, _] => (leading.iter().product(), *rows, 1),
            [] => return Err("a scalar has no rows".into()),
        };
        Ok(Self {
            matrices: to_usize(matrices)?,
            rows: to_usize(rows)?,
            axis_from_end,
        })
    }

    /// The bytes of a row of `plane`, a row of whose covers `per` of the
    /// tensor's rows: its every matrix's rows' bytes, divided evenly.
    fn row_bytes(&self, plane: &[u8], per: usize) -> usize {
        let rows = self.matrices * self.rows / per;
        let bytes = plane.len().checked_div(rows).unwrap_or(0);
        assert_eq!(bytes * rows, plane.len(), "a plane is whole rows");
        bytes
    }

    /// Rows `start..start + count` of each matrix of `plane`, a row of
    /// whose covers `per` of the tensor's rows, which the range starts and
    /// ends on ([`admit_transforms`]).
    fn select(&self, plane: &[u8], per: usize, start: usize, count: usize) -> Vec<u8> {
        assert!(
            start.is_multiple_of(per) && count.is_multiple_of(per),
            "an admitted range of rows starts and ends on a plane's rows"
        );
        let row_bytes = self.row_bytes(plane, per);
        let (rows, start, count) = (self.rows / per, start / per, count / per);
        let mut selected = Vec::with_capacity(self.matrices * count * row_bytes);
        for matrix in 0..self.matrices {
            let first = (matrix * rows + start) * row_bytes;
            selected.extend_from_slice(&plane[first..first + count * row_bytes]);
        }
        selected
    }

    /// `plane`'s rows, each a row of the tensor, each group of
    /// `order.len()` of them in `order`.
    fn permute(&self, plane: &[u8], order: &[usize]) -> Vec<u8> {
        let row_bytes = self.row_bytes(plane, 1);
        let period = order.len();
        let mut permuted = Vec::with_capacity(plane.len());
        for group in 0..self.matrices * self.rows / period {
            for &stored_row in order {
                let first = (group * period + stored_row) * row_bytes;
                permuted.extend_from_slice(&plane[first..first + row_bytes]);
            }
        }
        permuted
    }
}

fn to_usize(value: u64) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| "tensor extent exceeds the host address range".into())
}

/// The F32 little-endian values of a `shape` tensor stored packed as
/// `source` (`bytes`): the registered conversion into the canonical packet
/// representation, then its reference decode. Exact: a packed value (a
/// scale times a code) is an F32 value.
pub(crate) fn dequantize(source: Element, shape: &[u64], bytes: &[u8]) -> Result<Vec<u8>, String> {
    let packet = crate::planning::value_packets(source)
        .ok_or_else(|| format!("{} has no packet representation", source.name()))?;
    let canonical = packet
        .repack_host(source, shape, bytes)
        .ok_or_else(|| format!("{} has no conversion into {}", source.name(), packet.name()))?;
    let values = packet
        .decode_host(shape, &canonical)
        .ok_or_else(|| format!("{} does not decode on the host", packet.name()))?;
    Ok(values
        .into_iter()
        .flat_map(|value| (value as f32).to_le_bytes())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_family_contracts::RowRange;

    fn descriptor(shape: &[u64], transforms: Vec<ImportTransform>) -> WeightDescriptor {
        WeightDescriptor {
            name: "w".into(),
            shape: shape.to_vec(),
            transforms,
        }
    }

    fn f32s(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    #[test]
    fn row_ranges_select_rows_of_every_stacked_matrix() {
        // Two stacked [3, 2] matrices; keep rows 1..3 of each.
        let stored = [2, 3, 2];
        let transforms = vec![ImportTransform::Rows(RowRange { start: 1, rows: 2 })];
        let weight = descriptor(&[2, 2, 2], transforms);
        assert_eq!(admit(&weight, &stored, Element::f32()).unwrap(), [2, 2, 2]);
        let values = (0..12).map(|value| value as f32).collect::<Vec<_>>();
        let out = apply(&weight, &stored, f32s(&values), None).unwrap();
        assert_eq!(out, f32s(&[2., 3., 4., 5., 8., 9., 10., 11.]));
    }

    #[test]
    fn permutations_reorder_rows_within_each_group_and_move_packed_rows_whole() {
        let stored = [4, 32];
        let weight = descriptor(
            &[4, 32],
            vec![ImportTransform::PermuteRows { order: vec![1, 0] }],
        );
        let q8 = Element::named("gguf_q8_0").unwrap();
        admit(&weight, &stored, q8).unwrap();
        // One Q8_0 block (34 bytes) per row; rows tagged by their first byte.
        let bytes = (0..4u8).flat_map(|row| vec![row; 34]).collect::<Vec<_>>();
        let out = apply(&weight, &stored, bytes, None).unwrap();
        let tags = out.chunks(34).map(|row| row[0]).collect::<Vec<_>>();
        assert_eq!(tags, [1, 0, 3, 2]);
    }

    #[test]
    fn scales_multiply_f32_once_and_reject_other_storage() {
        let weight = descriptor(&[2], vec![ImportTransform::Scale { factor: 0.5 }]);
        assert!(admit(&weight, &[2], Element::bf16()).is_err());
        admit(&weight, &[2], Element::f32()).unwrap();
        let out = apply(&weight, &[2], f32s(&[3.0, -1.0]), None).unwrap();
        assert_eq!(out, f32s(&[1.5, -0.5]));
    }

    #[test]
    fn flattening_keeps_bytes_and_second_level_scales_stay_apart_from_them() {
        let flat = descriptor(&[8], vec![ImportTransform::Flatten]);
        assert_eq!(admit(&flat, &[2, 4], Element::f32()).unwrap(), [8]);
        let table = descriptor(&[2, 4], vec![ImportTransform::DropUnitAxis]);
        assert_eq!(admit(&table, &[1, 2, 4], Element::f32()).unwrap(), [2, 4]);
        assert!(admit(&table, &[2, 2, 4], Element::f32()).is_err());
        let values = f32s(&[1., 2., 3., 4., 5., 6., 7., 8.]);
        assert_eq!(
            apply(&table, &[1, 2, 4], values.clone(), None).unwrap(),
            values
        );
        let scale = || ImportTransform::ScaleByTensor {
            tensor: "w.scale".into(),
            activations: ScaledActivations::Rounded,
        };
        let scaled = descriptor(&[2, 32], vec![scale()]);
        let q8 = Element::named("gguf_q8_0").unwrap();
        assert_eq!(admit(&scaled, &[2, 32], q8).unwrap(), [2, 32]);
        assert_eq!(scale_tensor(&scaled), Some("w.scale"));
        let bytes = vec![7u8; 68];
        assert_eq!(apply(&scaled, &[2, 32], bytes.clone(), None).unwrap(), bytes);
        // A flattened matrix has no matrices to scale; one scale per weight.
        let flattened = descriptor(&[64], vec![scale(), ImportTransform::Flatten]);
        assert!(admit(&flattened, &[2, 32], Element::f32()).is_err());
        let twice = descriptor(
            &[2, 32],
            vec![
                scale(),
                ImportTransform::ScaleByTensor {
                    tensor: "w.other".into(),
                    activations: ScaledActivations::Rounded,
                },
            ],
        );
        assert!(admit(&twice, &[2, 32], q8).is_err());
    }

    /// Under a grid a weight's bytes are planar, as the file stores them:
    /// each matrix's codes, then its grid, moved as read. A range of rows
    /// takes each matrix's rows of codes and the grid's rows of their
    /// tiles, and so starts and ends on tile rows, or is refused.
    #[test]
    fn a_gridded_weight_is_planar_and_its_row_ranges_take_whole_tile_rows() {
        let grid = |tile| ImportTransform::ScaleBlocks { tensor: "w.scale".into(), tile, codes: ScaleCodes::Ue8m0 };
        // Two [2, 64] matrices under [1, 32] tiles: each row's codes its
        // index, each tile's scale its own.
        let stack = descriptor(&[2, 2, 64], vec![grid([1, 32])]);
        let source = Element::named("gguf_f8_e4m3_e8m0_1x32").unwrap();
        admit(&stack, &[2, 2, 64], source).unwrap();
        let codes = (0..4u8).flat_map(|row| [row; 64]).collect::<Vec<_>>();
        let scales = (100..108u8).collect::<Vec<_>>();
        let planar = apply(&stack, &[2, 2, 64], codes.clone(), Some(&scales)).unwrap();
        assert_eq!(planar, [&codes[..128], &scales[..4], &codes[128..], &scales[4..]].concat());
        assert_eq!(planar.len() as u64, source.canonical_byte_len(&[2, 2, 64]).unwrap());
        // A [96, 32] matrix under [32, 32] tiles: rows 32..96 are its codes'
        // and its grid's tile rows 1..3.
        let rows = |start, rows| {
            descriptor(&[rows, 32], vec![grid([32, 32]), ImportTransform::Rows(RowRange { start, rows })])
        };
        let source = Element::named("gguf_f8_e4m3_e8m0_32x32").unwrap();
        let codes = (0..96u8).flat_map(|row| [row; 32]).collect::<Vec<_>>();
        let scales = [7u8, 8, 9];
        let taken = rows(32, 64);
        assert_eq!(admit(&taken, &[96, 32], source).unwrap(), [64, 32]);
        assert_eq!(apply(&taken, &[96, 32], codes.clone(), Some(&scales)).unwrap(), [&codes[32 * 32..], &scales[1..]].concat());
        let error = admit(&rows(16, 32), &[96, 32], source).unwrap_err();
        assert!(error.contains("tiles of 32 rows"), "{error}");
    }

    /// A Q8_0 block (an f16 scale, then 32 int8 codes) dequantizes to the
    /// scale times each code, exactly.
    #[test]
    fn q8_0_dequantizes_exactly_to_f32() {
        let mut bytes = 0x3800_u16.to_le_bytes().to_vec(); // 0.5
        bytes.extend((-16i8..16).map(|code| code as u8));
        let source = Element::named("gguf_q8_0").unwrap();
        let values = dequantize(source, &[1, 32], &bytes).unwrap();
        let values = values
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(
            values,
            (-16..16).map(|code| code as f32 * 0.5).collect::<Vec<_>>()
        );
    }
}
