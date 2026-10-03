//! Exact import-time operations on stored tensors (model-family plan §2.3).
//!
//! A family names them in `WeightDescriptor::transforms`. They run on the
//! host over the stored bytes before the ordinary dense or repack import, so
//! the device importers only ever see a plain tensor of the logical shape.
//! Row operations move whole encoded rows and are exact for every encoding;
//! a scale multiplies F32 values once in F32. A second-level scale
//! (`ScaleByTensor`) leaves the stored bytes as they are: it becomes resident
//! beside the weight (`WeightPlan::scale`) and entries apply it to their F32
//! accumulator.

use magnitude_artifacts::gguf::Encoding;
use magnitude_family_contracts::{ImportTransform, WeightDescriptor};
use seismic::Element;

/// Whether the importer can apply `descriptor`'s transforms to a tensor
/// stored as `encoding` with `stored` shape, and the logical shape they
/// produce. Checked at plan time, so an unsupported transform makes the
/// model's plan fail rather than its load.
pub(crate) fn admit(
    descriptor: &WeightDescriptor,
    stored: &[u64],
    encoding: Encoding,
) -> Result<Vec<u64>, String> {
    let dense = matches!(encoding, Encoding::F32 | Encoding::F16 | Encoding::BF16);
    let mut shape = stored.to_vec();
    for transform in &descriptor.transforms {
        match transform {
            ImportTransform::Rows(_) | ImportTransform::PermuteRows { .. }
                if shape.len() == 1 && !dense =>
            {
                return Err(format!(
                    "{:?}: element rows of a packed vector cannot be selected",
                    descriptor.name
                ))
            }
            ImportTransform::Scale { .. } if encoding != Encoding::F32 => {
                return Err(format!(
                    "{:?}: a scale is exact only on F32 storage, not {encoding:?}",
                    descriptor.name
                ))
            }
            ImportTransform::ScaleByTensor { tensor }
                if descriptor
                    .transforms
                    .iter()
                    .filter(|other| {
                        matches!(
                            other,
                            ImportTransform::Flatten | ImportTransform::ScaleByTensor { .. }
                        )
                    })
                    .count()
                    > 1 =>
            {
                return Err(format!(
                    "{:?}: second-level scale tensor {tensor:?} must be the weight's only scale \
                     of its matrices",
                    descriptor.name
                ))
            }
            ImportTransform::Progressive(_)
                if encoding != Encoding::Q8_0 || descriptor.transforms.len() != 1 =>
            {
                return Err(format!(
                    "{:?}: a progressive plane is the only transform of a Q8_0 matrix",
                    descriptor.name
                ))
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
    if shape != descriptor.shape {
        return Err(format!(
            "{:?}: stored shape {stored:?} transforms to {shape:?}, not {:?}",
            descriptor.name, descriptor.shape
        ));
    }
    Ok(shape)
}

/// The logical source bytes of a tensor stored as `source` with `stored`
/// shape, after `descriptor`'s transforms (admitted by [`admit`]).
pub(crate) fn apply(
    descriptor: &WeightDescriptor,
    stored: &[u64],
    source: Element,
    mut bytes: Vec<u8>,
) -> Result<Vec<u8>, String> {
    let mut shape = stored.to_vec();
    for transform in &descriptor.transforms {
        match transform {
            ImportTransform::Rows(range) => {
                let rows = RowLayout::of(&shape, source)?;
                let start = to_usize(range.start)?;
                let count = to_usize(range.rows)?;
                let mut selected = Vec::with_capacity(rows.matrices * count * rows.row_bytes);
                for matrix in 0..rows.matrices {
                    let first = (matrix * rows.rows + start) * rows.row_bytes;
                    selected.extend_from_slice(&bytes[first..first + count * rows.row_bytes]);
                }
                bytes = selected;
                *shape
                    .iter_mut()
                    .rev()
                    .nth(rows.axis_from_end)
                    .expect("the row axis exists") = range.rows;
            }
            ImportTransform::PermuteRows { order } => {
                let rows = RowLayout::of(&shape, source)?;
                let period = order.len();
                let mut permuted = Vec::with_capacity(bytes.len());
                for group in 0..rows.matrices * rows.rows / period {
                    for &stored_row in order {
                        let first = (group * period + to_usize(stored_row)?) * rows.row_bytes;
                        permuted.extend_from_slice(&bytes[first..first + rows.row_bytes]);
                    }
                }
                bytes = permuted;
            }
            ImportTransform::Flatten => shape = vec![shape.iter().product()],
            ImportTransform::Scale { factor } => {
                let factor = *factor as f32;
                for word in bytes.chunks_exact_mut(4) {
                    let value = f32::from_le_bytes(word.try_into().expect("four bytes"));
                    word.copy_from_slice(&(value * factor).to_le_bytes());
                }
            }
            // Resident beside the weight, not applied to its bytes.
            ImportTransform::ScaleByTensor { .. } => {}
            ImportTransform::Progressive(_) => {
                return Err(format!(
                    "{:?}: a progressive plane is placed by `upload_bytes`",
                    descriptor.name
                ))
            }
        }
    }
    Ok(bytes)
}

/// The bytes the importer uploads for a weight stored as `source` (packed as
/// `encoding`, or dense) with `stored` shape: the transformed bytes in
/// `upload`, the plan's upload representation. A packed weight whose upload
/// differs from its source is dequantized exactly to F32; a progressive
/// plane is placed from the stored Q8_0 rows (`crate::progressive`).
pub(crate) fn upload_bytes(
    descriptor: &WeightDescriptor,
    stored: &[u64],
    source: Element,
    encoding: Option<Encoding>,
    upload: Element,
    bytes: Vec<u8>,
) -> Result<Vec<u8>, String> {
    if let [ImportTransform::Progressive(plane)] = descriptor.transforms[..] {
        let (&[rows, columns], Some(Encoding::Q8_0)) = (stored, encoding) else {
            return Err(format!("{:?}: a progressive plane places a Q8_0 matrix", descriptor.name));
        };
        return Ok(crate::progressive::plane(plane, &bytes, to_usize(rows)?, to_usize(columns)?));
    }
    let transformed = apply(descriptor, stored, source, bytes)?;
    if upload == source {
        return Ok(transformed);
    }
    let encoding = encoding
        .ok_or_else(|| format!("{:?}: a dequantized weight is stored dense", descriptor.name))?;
    dequantize(encoding, source, &descriptor.shape, &transformed)
}

/// The stored tensor holding `descriptor`'s second-level scale, when it has
/// one (at most one, [`admit`]).
pub(crate) fn scale_tensor(descriptor: &WeightDescriptor) -> Option<&str> {
    descriptor.transforms.iter().find_map(|transform| match transform {
        ImportTransform::ScaleByTensor { tensor } => Some(tensor.as_str()),
        _ => None,
    })
}

/// How a tensor's bytes divide into the rows its transforms address.
struct RowLayout {
    /// Matrices of a stack (1 for a matrix or vector).
    matrices: usize,
    /// Rows of each matrix.
    rows: usize,
    row_bytes: usize,
    /// The row axis counted from the last axis.
    axis_from_end: usize,
}

impl RowLayout {
    fn of(shape: &[u64], source: Element) -> Result<Self, String> {
        let (matrices, rows, columns, axis_from_end) = match shape {
            [elements] => (1, *elements, 1, 0),
            [leading @ .., rows, columns] => (leading.iter().product(), *rows, *columns, 1),
            [] => return Err("a scalar has no rows".into()),
        };
        let row_bytes = if shape.len() == 1 {
            source
                .dtype()
                .ok_or("element rows of a packed vector cannot be moved")?
                .bytes() as u64
        } else {
            source
                .canonical_byte_len(&[1, columns])
                .map_err(|error| format!("{}: {error}", source.name()))?
        };
        Ok(Self {
            matrices: to_usize(matrices)?,
            rows: to_usize(rows)?,
            row_bytes: to_usize(row_bytes)?,
            axis_from_end,
        })
    }
}

fn to_usize(value: u64) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| "tensor extent exceeds the host address range".into())
}

/// The F32 little-endian values of a `shape` tensor stored packed as
/// `encoding` (`source` bytes): the registered conversion into the canonical
/// packet representation, then its reference decode. Exact: a packed value
/// (a scale times a code) is an F32 value.
pub(crate) fn dequantize(
    encoding: Encoding,
    source: Element,
    shape: &[u64],
    bytes: &[u8],
) -> Result<Vec<u8>, String> {
    let packet = crate::resident_element(encoding, seismic::DType::F32, seismic::Layout::Packet)
        .ok_or_else(|| format!("{encoding:?} has no packet representation"))?;
    let canonical = packet
        .repack_host(source, shape, bytes)
        .ok_or_else(|| format!("{encoding:?} has no conversion into {}", packet.name()))?;
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
        values.iter().flat_map(|value| value.to_le_bytes()).collect()
    }

    #[test]
    fn row_ranges_select_rows_of_every_stacked_matrix() {
        // Two stacked [3, 2] matrices; keep rows 1..3 of each.
        let stored = [2, 3, 2];
        let transforms = vec![ImportTransform::Rows(RowRange { start: 1, rows: 2 })];
        let weight = descriptor(&[2, 2, 2], transforms);
        assert_eq!(admit(&weight, &stored, Encoding::F32).unwrap(), [2, 2, 2]);
        let values = (0..12).map(|value| value as f32).collect::<Vec<_>>();
        let out = apply(&weight, &stored, Element::f32(), f32s(&values)).unwrap();
        assert_eq!(out, f32s(&[2., 3., 4., 5., 8., 9., 10., 11.]));
    }

    #[test]
    fn permutations_reorder_rows_within_each_group_and_move_packed_rows_whole() {
        let stored = [4, 32];
        let weight = descriptor(
            &[4, 32],
            vec![ImportTransform::PermuteRows {
                order: vec![1, 0],
            }],
        );
        let q8 = Element::named("gguf_q8_0").unwrap();
        admit(&weight, &stored, Encoding::Q8_0).unwrap();
        // One Q8_0 block (34 bytes) per row; rows tagged by their first byte.
        let bytes = (0..4u8).flat_map(|row| vec![row; 34]).collect::<Vec<_>>();
        let out = apply(&weight, &stored, q8, bytes).unwrap();
        let tags = out.chunks(34).map(|row| row[0]).collect::<Vec<_>>();
        assert_eq!(tags, [1, 0, 3, 2]);
    }

    #[test]
    fn scales_multiply_f32_once_and_reject_other_storage() {
        let weight = descriptor(&[2], vec![ImportTransform::Scale { factor: 0.5 }]);
        assert!(admit(&weight, &[2], Encoding::BF16).is_err());
        admit(&weight, &[2], Encoding::F32).unwrap();
        let out = apply(&weight, &[2], Element::f32(), f32s(&[3.0, -1.0])).unwrap();
        assert_eq!(out, f32s(&[1.5, -0.5]));
    }

    #[test]
    fn flattening_keeps_bytes_and_second_level_scales_stay_apart_from_them() {
        let flat = descriptor(&[8], vec![ImportTransform::Flatten]);
        assert_eq!(admit(&flat, &[2, 4], Encoding::F32).unwrap(), [8]);
        let scale = || ImportTransform::ScaleByTensor {
            tensor: "w.scale".into(),
        };
        let scaled = descriptor(&[2, 32], vec![scale()]);
        assert_eq!(admit(&scaled, &[2, 32], Encoding::Q8_0).unwrap(), [2, 32]);
        assert_eq!(scale_tensor(&scaled), Some("w.scale"));
        let bytes = vec![7u8; 68];
        let q8 = Element::named("gguf_q8_0").unwrap();
        assert_eq!(apply(&scaled, &[2, 32], q8, bytes.clone()).unwrap(), bytes);
        // A flattened matrix has no matrices to scale; one scale per weight.
        let flattened = descriptor(&[64], vec![scale(), ImportTransform::Flatten]);
        assert!(admit(&flattened, &[2, 32], Encoding::F32).is_err());
        let twice = descriptor(
            &[2, 32],
            vec![
                scale(),
                ImportTransform::ScaleByTensor {
                    tensor: "w.other".into(),
                },
            ],
        );
        assert!(admit(&twice, &[2, 32], Encoding::Q8_0).is_err());
    }

    /// A Q8_0 block (an f16 scale, then 32 int8 codes) dequantizes to the
    /// scale times each code, exactly.
    #[test]
    fn q8_0_dequantizes_exactly_to_f32() {
        let mut bytes = 0x3800_u16.to_le_bytes().to_vec(); // 0.5
        bytes.extend((-16i8..16).map(|code| code as u8));
        let source = Element::named("gguf_q8_0").unwrap();
        let values = dequantize(Encoding::Q8_0, source, &[1, 32], &bytes).unwrap();
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
