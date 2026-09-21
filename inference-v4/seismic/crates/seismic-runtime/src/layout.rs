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
