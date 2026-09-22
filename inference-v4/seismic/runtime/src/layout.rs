//! The canonical layout of one representation over given extents: the
//! layout host transfers use and the layout result tensors are reported
//! in. Derived from the registry's representation facts; owned nowhere
//! else in this crate.

use seismic_compiler::errors::ExecutionError;
use seismic_lang::ids::RepresentationId;
use seismic_lang::registry::{representation_info, RepresentationKind};

/// Canonical storage-unit strides and total byte length. Dense storage units
/// are elements; a packed representation's final-axis unit is one packet.
pub(crate) struct Layout {
    pub strides: Vec<u64>,
    pub byte_len: u64,
    pub alignment: u64,
    storage_unit_bytes: u64,
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
                storage_unit_bytes: u64::from(dtype.bytes()),
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
        storage_unit_bytes: u64::from(packet_size),
    })
}

/// Validates the exact canonical view geometry used at the call/workflow
/// boundary. A leading slice may shift the byte offset, but retains canonical
/// strides and has the canonical footprint for its sliced extents.
pub(crate) fn validates_view(
    representation: RepresentationId,
    extents: &[u64],
    strides: &[u64],
    byte_offset: u64,
    byte_len: u64,
) -> bool {
    let Ok(layout) = canonical(representation, extents) else {
        return false;
    };
    layout.strides == strides
        && layout.byte_len == byte_len
        && layout.storage_unit_bytes != 0
        && byte_offset % layout.storage_unit_bytes == 0
        && byte_offset.checked_add(byte_len).is_some()
}

pub(crate) struct LeadingSlice {
    pub relative_offset: u64,
    pub byte_len: u64,
    pub extents: Vec<u64>,
    pub strides: Vec<u64>,
}

pub(crate) fn leading_slice(
    representation: RepresentationId,
    extents: &[u64],
    strides: &[u64],
    start: u64,
    end: u64,
) -> Option<LeadingSlice> {
    let &leading = extents.first()?;
    if extents.len() != strides.len() || start > end || end > leading {
        return None;
    }
    let (relative_offset, byte_len) = match &representation_info(representation).kind {
        RepresentationKind::Dense(dtype) => {
            let row = strides[0].checked_mul(u64::from(dtype.bytes()))?;
            (start.checked_mul(row)?, (end - start).checked_mul(row)?)
        }
        RepresentationKind::Packed(packet) if extents.len() == 1 => {
            let group = u64::from(packet.group);
            if start % group != 0 || (end != leading && end % group != 0) {
                return None;
            }
            let first = start / group;
            let last = end.checked_add(group - 1)?.checked_div(group)?;
            let bytes = u64::from(packet.packet_size);
            (
                first.checked_mul(bytes)?,
                (last - first).checked_mul(bytes)?,
            )
        }
        RepresentationKind::Packed(packet) => {
            let row = strides[0].checked_mul(u64::from(packet.packet_size))?;
            (start.checked_mul(row)?, (end - start).checked_mul(row)?)
        }
        RepresentationKind::External(packet) if extents.len() == 1 => {
            let group = u64::from(packet.logical_group);
            if start % group != 0 || (end != leading && end % group != 0) {
                return None;
            }
            let first = start / group;
            let last = end.checked_add(group - 1)?.checked_div(group)?;
            let bytes = u64::from(packet.packet_size);
            (
                first.checked_mul(bytes)?,
                (last - first).checked_mul(bytes)?,
            )
        }
        RepresentationKind::External(packet) => {
            let row = strides[0].checked_mul(u64::from(packet.packet_size))?;
            (start.checked_mul(row)?, (end - start).checked_mul(row)?)
        }
    };
    let mut extents = extents.to_vec();
    extents[0] = end - start;
    Some(LeadingSlice {
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
    fn packet_axis_slices_require_packet_boundaries() {
        let representation = registry::representation("q8g32").unwrap();
        let layout = canonical(representation, &[65]).unwrap();
        assert_eq!(layout.strides, vec![1]);
        assert_eq!(layout.byte_len, 108);
        assert!(leading_slice(representation, &[65], &[1], 32, 65).is_some());
        assert!(leading_slice(representation, &[65], &[1], 1, 32).is_none());
        assert!(leading_slice(representation, &[65], &[1], 32, 33).is_none());
    }
}
