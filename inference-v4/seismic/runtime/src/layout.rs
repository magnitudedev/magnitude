//! The canonical layout of one representation over given extents: the
//! layout host transfers use, plus actual affine result footprints. Both derive
//! from the registry's representation facts in this owner.

use crate::api::kernel::ViewOperation;
use crate::api::TensorError;
use seismic_compiler::errors::ExecutionError;
use seismic_lang::ids::RepresentationId;
use seismic_lang::registry::{representation_info, RepresentationKind};

/// Canonical storage-unit strides and total byte length. Dense storage units
/// are elements; a packed representation's final-axis unit is one packet.
pub(crate) struct Layout {
    pub strides: Vec<u64>,
    pub byte_len: u64,
    pub alignment: u64,
}

/// The canonical representation layout. A size beyond `u64` is a real
/// allocation failure: no device addresses it.
pub(crate) fn canonical(
    representation: RepresentationId,
    extents: &[u64],
) -> Result<Layout, ExecutionError> {
    let info = representation_info(representation);
    match &info.kind {
        RepresentationKind::Dense(dtype) => {
            let (strides, logical_elements) =
                row_major(extents).ok_or_else(|| unaddressable(info.name, extents))?;
            let byte_len = logical_elements
                .checked_mul(u64::from(dtype.bytes()))
                .ok_or_else(|| unaddressable(info.name, extents))?;
            Ok(Layout {
                strides,
                byte_len,
                alignment: u64::from(dtype.bytes()),
            })
        }
        RepresentationKind::Packed(packet) => packet_layout(
            info.name,
            extents,
            packet.group,
            packet.packet_size,
            packet.packet_alignment,
        ),
        RepresentationKind::External(packet) => packet_layout(
            info.name,
            extents,
            packet.logical_group,
            packet.packet_size,
            packet.packet_alignment,
        ),
        RepresentationKind::PackedRows(rows) => {
            let too_low_rank = || {
                ExecutionError::AllocationFailed(format!(
                    "`{}` storage requires rank at least {}",
                    info.name,
                    if rows.tile_rows() > 1 { 2 } else { 1 }
                ))
            };
            let units = rows.storage_units(extents).ok_or_else(|| {
                if extents.len() < 2 {
                    too_low_rank()
                } else {
                    unaddressable(info.name, extents)
                }
            })?;
            let (strides, _) =
                row_major(&units).ok_or_else(|| unaddressable(info.name, extents))?;
            let byte_len = rows
                .bytes(extents)
                .ok_or_else(|| unaddressable(info.name, extents))?;
            Ok(Layout {
                strides,
                byte_len,
                alignment: seismic_lang::registry::ROW_ALIGNMENT,
            })
        }
    }
}

fn packet_layout(
    name: &str,
    extents: &[u64],
    logical_group: u32,
    packet_size: u32,
    packet_alignment: u32,
) -> Result<Layout, ExecutionError> {
    let (&logical_extent, outer) = extents.split_last().ok_or_else(|| {
        ExecutionError::AllocationFailed(format!(
            "packet representation `{name}` requires a packing axis"
        ))
    })?;
    let mut units = extents.to_vec();
    *units.last_mut().expect("packet rank was checked") =
        logical_extent.div_ceil(u64::from(logical_group));
    let (strides, _) = row_major(&units).ok_or_else(|| unaddressable(name, extents))?;
    let outer_rows = outer
        .iter()
        .try_fold(1u64, |rows, extent| rows.checked_mul(*extent))
        .ok_or_else(|| unaddressable(name, extents))?;
    let byte_len = outer_rows
        .checked_mul(logical_extent.div_ceil(u64::from(logical_group)))
        .and_then(|packets| packets.checked_mul(u64::from(packet_size)))
        .ok_or_else(|| unaddressable(name, extents))?;
    Ok(Layout {
        strides,
        byte_len,
        alignment: u64::from(packet_alignment),
    })
}

/// Visit canonical host storage bytes in logical order and their corresponding
/// affine device ranges. Reads and writes use this same representation map.
pub(crate) fn transfer_ranges(
    representation: RepresentationId,
    extents: &[u64],
    strides: &[u64],
    mut transfer: impl FnMut(u64, std::ops::Range<usize>) -> Result<(), ExecutionError>,
) -> Result<(), ExecutionError> {
    let invalid = || unaddressable(representation_info(representation).name, extents);
    let canonical = canonical(representation, extents)?;
    let host_len = usize::try_from(canonical.byte_len).map_err(|_| invalid())?;
    if host_len == 0 {
        return Ok(());
    }
    seismic_ir::storage::addressed_span_u64(representation, extents, strides)
        .ok_or_else(invalid)?;
    if strides == canonical.strides {
        return transfer(0, 0..host_len);
    }
    let (units, width) =
        seismic_ir::storage::concrete_storage_units(representation, extents).ok_or_else(invalid)?;
    let host_width = usize::try_from(width).map_err(|_| invalid())?;
    for linear in 0..canonical.byte_len / width {
        let mut remaining = linear;
        let mut storage = 0_u64;
        for axis in (0..units.len()).rev() {
            let coordinate = remaining % units[axis];
            remaining /= units[axis];
            storage = storage
                .checked_add(coordinate.checked_mul(strides[axis]).ok_or_else(invalid)?)
                .ok_or_else(invalid)?;
        }
        let offset = storage.checked_mul(width).ok_or_else(invalid)?;
        let host = usize::try_from(linear)
            .map_err(|_| invalid())?
            .checked_mul(host_width)
            .ok_or_else(invalid)?;
        transfer(offset, host..host + host_width)?;
    }
    Ok(())
}

/// Validate the actual descriptor footprint, retaining noncanonical affine
/// strides. Buffer-capacity and access-permission checks belong to its binding.
pub(crate) fn validates_view(
    representation: RepresentationId,
    extents: &[u64],
    strides: &[u64],
    byte_offset: u64,
    byte_len: u64,
) -> bool {
    seismic_ir::storage::valid_concrete_view(
        representation,
        extents,
        strides,
        byte_offset,
        byte_len,
    )
}

/// A tensor view's geometry: logical extents, storage-unit strides and the
/// byte range it addresses in its backing.
pub(crate) struct ViewGeometry {
    pub extents: Vec<u64>,
    pub strides: Vec<u64>,
    pub byte_offset: u64,
    pub byte_len: u64,
}

/// The one meaning of a [`ViewOperation`] over a tensor view. Tensor handles,
/// native-graph description and workflow planning all apply view operations
/// through this owner, so they accept and refuse exactly the same views.
pub(crate) fn apply_view(
    representation: RepresentationId,
    source: ViewGeometry,
    operation: &ViewOperation,
) -> Result<ViewGeometry, TensorError> {
    match operation {
        ViewOperation::LeadingSlice { start, end } => {
            let slice = leading_slice(
                representation,
                &source.extents,
                &source.strides,
                *start,
                *end,
            )?;
            let byte_offset = source
                .byte_offset
                .checked_add(slice.relative_offset)
                .ok_or_else(|| {
                    TensorError::Execution(unaddressable(
                        representation_info(representation).name,
                        &slice.extents,
                    ))
                })?;
            Ok(ViewGeometry {
                extents: slice.extents,
                strides: slice.strides,
                byte_offset,
                byte_len: slice.byte_len,
            })
        }
        // A reshape reinterprets contiguous row-major storage; a strided view
        // has no reshape that is still a view of the same elements.
        ViewOperation::Reshape { extents } => {
            if let RepresentationKind::PackedRows(rows) = &representation_info(representation).kind
            {
                let geometry = |extents: &[u64]| {
                    let tail = if rows.tile_rows() > 1 { 2 } else { 1 };
                    extents
                        .len()
                        .checked_sub(tail)
                        .map(|start| extents[start..].to_vec())
                };
                if geometry(extents).is_none() || geometry(extents) != geometry(&source.extents) {
                    return Err(TensorError::RowLayoutReshape {
                        representation: representation_info(representation).name,
                        extents: extents.clone(),
                    });
                }
            }
            if canonical(representation, &source.extents)?.strides != source.strides {
                return Err(TensorError::ReshapeLayout {
                    extents: source.extents,
                    strides: source.strides,
                });
            }
            let layout = canonical(representation, extents)?;
            if layout.byte_len != source.byte_len {
                return Err(TensorError::ReshapeStorage {
                    current_bytes: source.byte_len,
                    requested_bytes: layout.byte_len,
                });
            }
            Ok(ViewGeometry {
                extents: extents.clone(),
                strides: layout.strides,
                byte_offset: source.byte_offset,
                byte_len: source.byte_len,
            })
        }
    }
}

struct LeadingSlice {
    relative_offset: u64,
    byte_len: u64,
    extents: Vec<u64>,
    strides: Vec<u64>,
}

fn leading_slice(
    representation: RepresentationId,
    extents: &[u64],
    strides: &[u64],
    start: u64,
    end: u64,
) -> Result<LeadingSlice, TensorError> {
    let info = representation_info(representation);
    let leading = extents.first().copied().unwrap_or(0);
    if extents.is_empty() || start > end || end > leading {
        return Err(TensorError::SliceOutOfBounds {
            extent: leading,
            start,
            end,
        });
    }
    assert_eq!(
        extents.len(),
        strides.len(),
        "a tensor view has one stride per axis"
    );
    let (group, width) = match &info.kind {
        RepresentationKind::Dense(dtype) => (1, u64::from(dtype.bytes())),
        RepresentationKind::Packed(packet) => (packet.group, u64::from(packet.packet_size)),
        RepresentationKind::External(packet) => {
            (packet.logical_group, u64::from(packet.packet_size))
        }
        RepresentationKind::PackedRows(rows) => {
            // Rank 1 would slice the packing axis; an `mma16` row axis slices
            // in whole tiles.
            let tiled = rows.tile_rows() > 1 && extents.len() == 2;
            let tile = rows.tile_rows();
            if extents.len() == 1
                || (tiled && (start % tile != 0 || (end != leading && end % tile != 0)))
            {
                return Err(TensorError::UnalignedRowSlice {
                    representation: info.name,
                    start,
                    end,
                });
            }
            let stride = rows
                .row_stride_bytes(*extents.last().expect("row slice rank was checked"))
                .ok_or_else(|| TensorError::Execution(unaddressable(info.name, extents)))?;
            (1, stride)
        }
    };
    // Only a rank-1 packed tensor slices its packing axis.
    let packet_axis = extents.len() == 1 && !matches!(info.kind, RepresentationKind::Dense(_));
    let unit = u64::from(group);
    if packet_axis && (start % unit != 0 || (end != leading && end % unit != 0)) {
        return Err(TensorError::UnalignedPacketSlice { group, start, end });
    }
    let start_unit = if packet_axis { start / unit } else { start };
    let mut extents = extents.to_vec();
    extents[0] = end - start;
    let overflow = || TensorError::Execution(unaddressable(info.name, &extents));
    let (byte_len, _) = seismic_ir::storage::addressed_span_u64(representation, &extents, strides)
        .ok_or_else(overflow)?;
    // Empty views have no addressed element. Retain the backing's valid anchor
    // instead of inventing an out-of-allocation one-past strided coordinate.
    let relative_offset = if byte_len == 0 {
        0
    } else {
        start_unit
            .checked_mul(strides[0])
            .and_then(|units| units.checked_mul(width))
            .ok_or_else(overflow)?
    };
    Ok(LeadingSlice {
        relative_offset,
        byte_len,
        extents,
        strides: strides.to_vec(),
    })
}

fn row_major(extents: &[u64]) -> Option<(Vec<u64>, u64)> {
    let mut strides = vec![0u64; extents.len()];
    let mut elements = 1u64;
    for axis in (0..extents.len()).rev() {
        strides[axis] = elements;
        elements = elements.checked_mul(extents[axis])?;
    }
    Some((strides, elements))
}

fn unaddressable(name: &str, extents: &[u64]) -> ExecutionError {
    ExecutionError::AllocationFailed(format!(
        "a `{name}` tensor of extents {extents:?} exceeds the addressable byte range"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::registry;
    use seismic_lang::types::DType;

    #[test]
    fn dense_geometry_and_leading_slices_share_one_owner() {
        let representation = registry::dense(DType::F32);
        let layout = canonical(representation, &[4, 8]).unwrap();
        assert_eq!(layout.strides, vec![8, 1]);
        assert_eq!(layout.byte_len, 128);
        assert!(validates_view(representation, &[4, 8], &[8, 1], 0, 128));
        assert!(!validates_view(representation, &[4, 8], &[8, 2], 0, 128));

        let slice = leading_slice(representation, &[4, 8], &[8, 1], 1, 3).unwrap();
        assert_eq!(slice.relative_offset, 32);
        assert_eq!(slice.byte_len, 64);
        assert_eq!(slice.extents, vec![2, 8]);
        assert!(validates_view(
            representation,
            &slice.extents,
            &slice.strides,
            slice.relative_offset,
            slice.byte_len
        ));
    }

    #[test]
    fn strided_publication_and_leading_slice_use_addressed_span() {
        let repr = registry::dense(DType::U32);
        assert!(validates_view(repr, &[3], &[2], 4, 20));
        assert!(!validates_view(repr, &[3], &[2], 4, 12));
        let slice = leading_slice(repr, &[3, 2], &[1, 3], 1, 3).unwrap();
        assert_eq!(slice.relative_offset, 4);
        assert_eq!(slice.byte_len, 20);
        assert_eq!(slice.extents, vec![2, 2]);
        assert!(validates_view(
            repr,
            &slice.extents,
            &slice.strides,
            slice.relative_offset,
            slice.byte_len
        ));
        let empty = leading_slice(repr, &[3], &[10], 3, 3).unwrap();
        assert_eq!((empty.relative_offset, empty.byte_len), (0, 0));
        assert!(!validates_view(repr, &[u64::MAX], &[u64::MAX], 0, 4));
    }

    #[test]
    fn host_transfer_preserves_logical_order_and_storage_gaps() {
        let repr = registry::dense(DType::U32);
        let source = [1_u32, 2, 3, 4, 5, 6]
            .into_iter()
            .flat_map(u32::to_ne_bytes)
            .collect::<Vec<_>>();
        let mut host = vec![0; 24];
        transfer_ranges(repr, &[3, 2], &[1, 3], |offset, range| {
            host[range.clone()]
                .copy_from_slice(&source[offset as usize..offset as usize + range.len()]);
            Ok(())
        })
        .unwrap();
        let words = host
            .chunks_exact(4)
            .map(|word| u32::from_ne_bytes(word.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(words, [1, 4, 2, 5, 3, 6]);
        let mut sparse = vec![0xff; 20];
        transfer_ranges(repr, &[3], &[2], |offset, range| {
            sparse[offset as usize..offset as usize + range.len()].copy_from_slice(&source[range]);
            Ok(())
        })
        .unwrap();
        assert_eq!(&sparse[4..8], &[0xff; 4]);
        assert_eq!(&sparse[12..16], &[0xff; 4]);
        assert_eq!(&sparse[16..20], &3_u32.to_ne_bytes());
    }

    #[test]
    fn packet_axis_slices_require_packet_boundaries() {
        let representation = registry::representation("q8g32").unwrap();
        let layout = canonical(representation, &[65]).unwrap();
        assert_eq!(layout.strides, vec![1]);
        assert_eq!(layout.byte_len, 108);
        assert!(leading_slice(representation, &[65], &[1], 32, 65).is_ok());
        for (start, end) in [(1, 32), (32, 33)] {
            assert_eq!(
                leading_slice(representation, &[65], &[1], start, end).err(),
                Some(TensorError::UnalignedPacketSlice {
                    group: 32,
                    start,
                    end
                })
            );
        }
    }

    #[test]
    fn reshape_is_a_view_only_of_contiguous_storage() {
        let representation = registry::dense(DType::F32);
        let geometry = |strides: Vec<u64>| ViewGeometry {
            extents: vec![2, 3],
            strides,
            byte_offset: 8,
            byte_len: 24,
        };
        let reshaped = apply_view(
            representation,
            geometry(vec![3, 1]),
            &ViewOperation::Reshape {
                extents: vec![3, 2],
            },
        )
        .unwrap();
        assert_eq!(
            (reshaped.strides, reshaped.byte_offset, reshaped.byte_len),
            (vec![2, 1], 8, 24)
        );
        assert_eq!(
            apply_view(
                representation,
                geometry(vec![1, 2]),
                &ViewOperation::Reshape {
                    extents: vec![3, 2]
                },
            )
            .err(),
            Some(TensorError::ReshapeLayout {
                extents: vec![2, 3],
                strides: vec![1, 2]
            })
        );
        assert_eq!(
            apply_view(
                representation,
                geometry(vec![3, 1]),
                &ViewOperation::Reshape {
                    extents: vec![4, 2]
                },
            )
            .err(),
            Some(TensorError::ReshapeStorage {
                current_bytes: 24,
                requested_bytes: 32
            })
        );
        let sliced = apply_view(
            representation,
            geometry(vec![3, 1]),
            &ViewOperation::LeadingSlice { start: 1, end: 2 },
        )
        .unwrap();
        assert_eq!(
            (sliced.extents, sliced.byte_offset, sliced.byte_len),
            (vec![1, 3], 20, 12)
        );
    }
}
