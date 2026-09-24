//! Concrete (u64) geometry of one representation's storage.
use super::*;

/// Minimum alignment of one element of `representation`.
pub fn representation_alignment(representation: RepresentationId) -> u64 {
    match &registry::representation_info(representation).kind {
        RepresentationKind::Dense(dtype) => dtype.bytes() as u64,
        RepresentationKind::Packed(layout) => u64::from(layout.packet_alignment),
        RepresentationKind::PackedRows(_) => registry::ROW_ALIGNMENT,
        RepresentationKind::External(layout) => u64::from(layout.packet_alignment),
    }
}

/// Concrete per-axis storage units and the byte width of one unit.
/// Packed/external final axes count packets rather than logical elements.
/// Row layouts count rows: the final axis is one unit of the row stride, and
/// `mma16` pads the row axis to whole 16-row tiles.
pub fn concrete_storage_units(
    representation: RepresentationId,
    extents: &[u64],
) -> Option<(Vec<u64>, u64)> {
    let mut units = extents.to_vec();
    let width = match &registry::representation_info(representation).kind {
        RepresentationKind::Dense(dtype) => u64::from(dtype.bytes()),
        RepresentationKind::Packed(packet) => {
            *units.last_mut()? = extents.last()?.div_ceil(u64::from(packet.group));
            u64::from(packet.packet_size)
        }
        RepresentationKind::External(packet) => {
            *units.last_mut()? = extents.last()?.div_ceil(u64::from(packet.logical_group));
            u64::from(packet.packet_size)
        }
        RepresentationKind::PackedRows(layout) => {
            units = layout.storage_units(extents)?;
            layout.row_stride_bytes(*extents.last()?)?
        }
    };
    if width == 0 {
        return None;
    }
    Some((units, width))
}

/// Exact byte span and unit width for one concrete affine tensor descriptor.
/// An empty view addresses no storage, regardless of its strides.
pub fn addressed_span_u64(
    representation: RepresentationId,
    extents: &[u64],
    strides: &[u64],
) -> Option<(u64, u64)> {
    if extents.len() != strides.len() {
        return None;
    }
    let (units, width) = concrete_storage_units(representation, extents)?;
    if units.contains(&0) {
        return Some((0, width));
    }
    let last = units.iter().zip(strides).try_fold(0u64, |offset, (extent, stride)| {
        offset.checked_add((extent - 1).checked_mul(*stride)?)
    })?;
    Some((last.checked_add(1)?.checked_mul(width)?, width))
}

/// A public concrete view must describe exactly its addressed byte range.
/// The backing allocation's actual length is checked by its binding owner.
pub fn valid_concrete_view(
    representation: RepresentationId,
    extents: &[u64],
    strides: &[u64],
    byte_offset: u64,
    byte_len: u64,
) -> bool {
    let alignment = representation_alignment(representation);
    alignment != 0
        && byte_offset % alignment == 0
        && byte_offset.checked_add(byte_len).is_some()
        && addressed_span_u64(representation, extents, strides)
            .is_some_and(|(span, _)| span == byte_len)
}
