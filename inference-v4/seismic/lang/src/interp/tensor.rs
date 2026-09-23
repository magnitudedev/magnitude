//! Host storage used by the semantic oracle.

use crate::reference_math::{float_literal, ReferenceScalar};

use crate::ids::RepresentationId;
use crate::registry::{self, DecodeStep, PlaneEncoding, RepresentationKind};
use crate::types::DType;

#[derive(Clone, Debug)]
pub enum TensorData {
    Dense {
        representation: RepresentationId,
        shape: Vec<usize>,
        bytes: Vec<u8>,
        initialized: Vec<bool>,
    },
    /// Canonical packet-interleaved bytes for a packed resident or external
    /// representation.
    Encoded {
        representation: RepresentationId,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    },
}

impl TensorData {
    /// Payload allocated for oracle storage. Dense elements retain their
    /// representation's bits and one initialization byte per element.
    pub fn allocation_bytes(
        representation: RepresentationId,
        shape: &[usize],
    ) -> Result<u64, String> {
        let count = shape
            .iter()
            .try_fold(1usize, |n, x| n.checked_mul(*x))
            .ok_or("tensor size overflow")?;
        let data = match &registry::representation_info(representation).kind {
            RepresentationKind::Dense(dtype) => count
                .checked_mul(dtype.bytes() as usize + std::mem::size_of::<bool>())
                .ok_or("tensor size overflow")?,
            _ => encoded_bytes(representation, shape)?,
        };
        data.checked_add(
            shape
                .len()
                .checked_mul(std::mem::size_of::<usize>())
                .ok_or("tensor rank overflow")?,
        )
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| "tensor size overflow".into())
    }
    /// Actual retained vector payload capacities, for caller-owned input budgets.
    pub fn storage_bytes(&self) -> Result<u64, String> {
        let (shape, data) = match self {
            Self::Dense {
                shape,
                bytes,
                initialized,
                ..
            } => (
                shape,
                bytes
                    .capacity()
                    .checked_add(initialized.capacity())
                    .ok_or("tensor size overflow")?,
            ),
            Self::Encoded { shape, bytes, .. } => (shape, bytes.capacity()),
        };
        data.checked_add(
            shape
                .capacity()
                .checked_mul(std::mem::size_of::<usize>())
                .ok_or("tensor rank overflow")?,
        )
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| "tensor size overflow".into())
    }

    pub fn dense(dtype: DType, shape: Vec<usize>, values: Vec<f64>) -> Self {
        assert_eq!(values.len(), shape.iter().product::<usize>());
        let mut data = Self::uninitialized(registry::dense(dtype), shape)
            .expect("dense tensor geometry is host-addressable");
        for (index, value) in values.into_iter().enumerate() {
            data.write(index, scalar_from_number(dtype, value))
                .expect("dense element index is in bounds");
        }
        data
    }

    /// Take ownership of canonical native bytes without a floating conversion.
    /// In particular, immutable NaN payloads and signed zero remain unchanged.
    pub fn dense_from_bytes(
        dtype: DType,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    ) -> Result<Self, String> {
        let count = element_count(&shape)?;
        let expected = count
            .checked_mul(dtype.bytes() as usize)
            .ok_or("tensor size overflow")?;
        if bytes.len() != expected {
            return Err(format!(
                "dense tensor has {} bytes; canonical layout requires {expected}",
                bytes.len()
            ));
        }
        Ok(Self::Dense {
            representation: registry::dense(dtype),
            initialized: vec![true; count],
            shape,
            bytes,
        })
    }

    pub fn encoded(
        representation: RepresentationId,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    ) -> Result<Self, String> {
        let expected = encoded_bytes(representation, &shape)?;
        if bytes.len() != expected {
            return Err(format!(
                "encoded tensor has {} bytes; canonical layout requires {expected}",
                bytes.len()
            ));
        }
        Ok(Self::Encoded {
            representation,
            shape,
            bytes,
        })
    }

    pub(super) fn uninitialized(
        representation: RepresentationId,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        match &registry::representation_info(representation).kind {
            RepresentationKind::Dense(dtype) => {
                let count = element_count(&shape)?;
                let bytes = count
                    .checked_mul(dtype.bytes() as usize)
                    .ok_or("tensor size overflow")?;
                Ok(Self::Dense {
                    representation,
                    shape,
                    bytes: vec![0; bytes],
                    initialized: vec![false; count],
                })
            }
            RepresentationKind::Packed(_) => Ok(Self::Encoded {
                representation,
                bytes: vec![0; encoded_bytes(representation, &shape)?],
                shape,
            }),
            RepresentationKind::External(_) => {
                Err("external artifact storage cannot be allocated by a kernel".to_owned())
            }
        }
    }

    pub fn representation(&self) -> RepresentationId {
        match self {
            Self::Dense { representation, .. } | Self::Encoded { representation, .. } => {
                *representation
            }
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Dense { shape, .. } | Self::Encoded { shape, .. } => shape,
        }
    }

    /// Decode the logical contents through the representation's registered
    /// reference recipe, in row-major order.
    pub fn values(&self) -> Result<Vec<f64>, String> {
        let count = self.shape().iter().product();
        let mut values = Vec::with_capacity(count);
        for index in 0..count {
            values.push(self.read(index)?);
        }
        Ok(values)
    }

    pub(super) fn read(&self, flat: usize) -> Result<f64, String> {
        self.read_scalar(flat).map(ReferenceScalar::to_f64)
    }

    pub(super) fn read_scalar(&self, flat: usize) -> Result<ReferenceScalar, String> {
        match self {
            Self::Dense { representation, .. } => {
                let RepresentationKind::Dense(dtype) =
                    registry::representation_info(*representation).kind
                else {
                    unreachable!("dense oracle storage has a non-dense representation")
                };
                Ok(read_dense(dtype, self.dense_element_bytes(flat)?))
            }
            Self::Encoded {
                representation,
                shape,
                bytes,
            } => decode(*representation, shape, bytes, flat),
        }
    }

    pub(super) fn write(&mut self, flat: usize, value: ReferenceScalar) -> Result<(), String> {
        match self {
            Self::Dense {
                representation,
                bytes,
                initialized,
                ..
            } => {
                let RepresentationKind::Dense(dtype) =
                    &registry::representation_info(*representation).kind
                else {
                    unreachable!("dense oracle storage has a non-dense representation")
                };
                let initialized = initialized
                    .get_mut(flat)
                    .ok_or("write outside tensor bounds")?;
                let width = dtype.bytes() as usize;
                let value = super::scalar::cast(*dtype, value);
                bytes[flat * width..(flat + 1) * width]
                    .copy_from_slice(&value.bits().to_le_bytes()[..width]);
                *initialized = true;
                Ok(())
            }
            Self::Encoded { .. } => Err("read-only representation cannot be written".to_owned()),
        }
    }

    pub(super) fn dense_element_bytes(&self, flat: usize) -> Result<&[u8], String> {
        let Self::Dense {
            representation,
            bytes,
            initialized,
            ..
        } = self
        else {
            return Err("dense element requested from encoded tensor".into());
        };
        match initialized.get(flat) {
            Some(true) => {}
            Some(false) => return Err("read of uninitialized tensor element".into()),
            None => return Err("read outside tensor bounds".into()),
        }
        let RepresentationKind::Dense(dtype) = registry::representation_info(*representation).kind
        else {
            unreachable!("dense oracle storage has a non-dense representation")
        };
        let width = dtype.bytes() as usize;
        Ok(&bytes[flat * width..(flat + 1) * width])
    }

    pub(super) fn canonical_bytes(&self) -> Result<&[u8], String> {
        match self {
            Self::Dense {
                bytes, initialized, ..
            } => {
                if initialized.iter().any(|initialized| !initialized) {
                    return Err("read of uninitialized tensor element".into());
                }
                Ok(bytes)
            }
            Self::Encoded { bytes, .. } => Ok(bytes),
        }
    }

    pub(super) fn encoded_parts(&self) -> Option<(RepresentationId, &[usize], &[u8])> {
        match self {
            Self::Encoded {
                representation,
                shape,
                bytes,
            } => Some((*representation, shape, bytes)),
            Self::Dense { .. } => None,
        }
    }
}

fn element_count(shape: &[usize]) -> Result<usize, String> {
    shape
        .iter()
        .try_fold(1usize, |count, extent| count.checked_mul(*extent))
        .ok_or_else(|| "tensor size overflow".into())
}

pub(super) fn encoded_bytes(
    representation: RepresentationId,
    shape: &[usize],
) -> Result<usize, String> {
    let (last, outer) = shape
        .split_last()
        .ok_or("encoded representation requires rank at least one")?;
    let rows = outer.iter().try_fold(1u64, |value, axis| {
        value
            .checked_mul(*axis as u64)
            .ok_or("tensor size overflow")
    })?;
    let bytes = match &registry::representation_info(representation).kind {
        RepresentationKind::Packed(layout) => layout
            .bytes(rows, *last as u64)
            .ok_or("packed tensor size overflow")?,
        RepresentationKind::External(layout) => rows
            .checked_mul((*last as u64).div_ceil(u64::from(layout.logical_group)))
            .and_then(|packets| packets.checked_mul(u64::from(layout.packet_size)))
            .ok_or("external tensor size overflow")?,
        RepresentationKind::Dense(dtype) => rows
            .checked_mul(*last as u64)
            .and_then(|elements| elements.checked_mul(u64::from(dtype.bytes())))
            .ok_or("dense tensor size overflow")?,
    };
    usize::try_from(bytes).map_err(|_| "tensor is not host-addressable".to_owned())
}

fn decode(
    representation: RepresentationId,
    shape: &[usize],
    bytes: &[u8],
    flat: usize,
) -> Result<ReferenceScalar, String> {
    let info = registry::representation_info(representation);
    let RepresentationKind::Packed(layout) = &info.kind else {
        return Err("external representation is conversion-only".to_owned());
    };
    let width = *shape.last().ok_or("packed tensor has rank zero")?;
    if flat >= shape.iter().product() {
        return Err("read outside tensor bounds".to_owned());
    }
    let row = flat / width;
    let column = flat % width;
    let packets_per_row = width.div_ceil(layout.group as usize);
    let packet = row * packets_per_row + column / layout.group as usize;
    let local = (column % layout.group as usize) as u64;
    let recipe = registry::decode_recipe(representation, info.decoded)
        .ok_or("packed representation has no decode recipe")?;
    let mut temporaries = vec![ReferenceScalar::U32(0); recipe.temporary_count()];
    for step in recipe.steps() {
        let value = match step {
            DecodeStep::ReadPlaneField { plane, field, .. } => {
                let schema = &recipe.planes()[*plane as usize];
                let entry = schema.entry(local, *field);
                read_plane(layout, bytes, packet, *plane as usize, entry)?
            }
            DecodeStep::InterpretCode {
                raw,
                bits,
                interpretation,
                ..
            } => ReferenceScalar::I32(
                interpretation.decode(temporaries[recipe.ordinal(*raw)].bits(), *bits),
            ),
            DecodeStep::DecodeFloatCode { raw, format, .. } => {
                let scalar = crate::reference_math::float_code_recipe(*format);
                crate::reference_math::evaluate(&scalar, &[temporaries[recipe.ordinal(*raw)]])
                    .expect("floating code interpretation is total")
            }
            DecodeStep::ConvertToF32 { from, .. } => {
                super::scalar::cast(DType::F32, temporaries[recipe.ordinal(*from)])
            }
            DecodeStep::Multiply { left, right, .. } => super::scalar::binary(
                crate::syntax::ast::BinaryOp::Mul,
                temporaries[recipe.ordinal(*left)],
                temporaries[recipe.ordinal(*right)],
                Some(DType::F32),
            )
            .expect("registered packed decode arithmetic is total"),
            DecodeStep::Negate { from, .. } => super::scalar::unary(
                crate::syntax::ast::UnaryOp::Neg,
                temporaries[recipe.ordinal(*from)],
            )
            .expect("registered packed decode arithmetic is total"),
            DecodeStep::MultiplyAdd {
                factor,
                multiplicand,
                addend,
                ..
            } => super::scalar::math(
                crate::intrinsics::MathOp::Fma,
                &[
                    temporaries[recipe.ordinal(*factor)],
                    temporaries[recipe.ordinal(*multiplicand)],
                    temporaries[recipe.ordinal(*addend)],
                ],
            )
            .expect("registered packed decode arithmetic is total"),
            DecodeStep::Cast { from, to, .. } => {
                super::scalar::cast(*to, temporaries[recipe.ordinal(*from)])
            }
        };
        let into = step.defines();
        temporaries[recipe.ordinal(into)] = value;
    }
    Ok(temporaries[recipe.ordinal(recipe.output())])
}

fn read_plane(
    layout: &crate::registry::PackedPacketLayout,
    bytes: &[u8],
    packet: usize,
    plane: usize,
    entry: u64,
) -> Result<ReferenceScalar, String> {
    let schema = layout
        .planes
        .get(plane)
        .ok_or("decode names unknown plane")?;
    let start = packet
        .checked_mul(layout.packet_size as usize)
        .and_then(|offset| offset.checked_add(schema.offset as usize))
        .ok_or("plane offset overflow")?;
    let plane_bytes = bytes
        .get(start..start + schema.bytes_per_group as usize)
        .ok_or("plane read outside encoded tensor")?;
    Ok(match &schema.encoding {
        PlaneEncoding::Packed { .. } | PlaneEncoding::FloatCode { .. } => {
            ReferenceScalar::U32(read_bits(
                plane_bytes,
                entry as usize * schema.entry_bits as usize,
                schema.entry_bits,
            ))
        }
        PlaneEncoding::Dense(dtype) => {
            let offset = entry as usize * dtype.bytes() as usize;
            read_dense(
                *dtype,
                &plane_bytes[offset..offset + dtype.bytes() as usize],
            )
        }
    })
}

pub(super) fn read_bits(bytes: &[u8], first: usize, width: u32) -> u32 {
    (0..width as usize).fold(0, |value, bit| {
        value | (u32::from((bytes[(first + bit) / 8] >> ((first + bit) % 8)) & 1) << bit)
    })
}

pub(super) fn write_bits(bytes: &mut [u8], first: usize, width: u32, value: u32) {
    for bit in 0..width as usize {
        let byte = (first + bit) / 8;
        let shift = (first + bit) % 8;
        bytes[byte] = (bytes[byte] & !(1 << shift)) | (((value >> bit) as u8 & 1) << shift);
    }
}

fn read_dense(dtype: DType, bytes: &[u8]) -> ReferenceScalar {
    let mut payload = [0; 4];
    payload[..bytes.len()].copy_from_slice(bytes);
    ReferenceScalar::from_bits(dtype, u32::from_le_bytes(payload))
}

pub(super) fn scalar_from_number(dtype: DType, value: f64) -> ReferenceScalar {
    if dtype.is_float() {
        float_literal(dtype, value)
    } else {
        match dtype {
            DType::I32 => ReferenceScalar::I32(value as i32),
            DType::U32 => ReferenceScalar::U32(value as u32),
            DType::Bool => ReferenceScalar::Bool(value != 0.0),
            _ => unreachable!(),
        }
    }
}

pub fn round_to(dtype: DType, value: f64) -> f64 {
    scalar_from_number(dtype, value).to_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_dense_bits_survive_reads_without_reencoding() {
        for (dtype, bytes) in [
            (
                DType::F32,
                [0x8000_0000u32, 0x7f80_0001, 0x7fc1_2345]
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>(),
            ),
            (
                DType::F16,
                [0x8000u16, 0x7c01, 0x7e45]
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect(),
            ),
            (
                DType::BF16,
                [0x8000u16, 0x7f81, 0x7fc5]
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect(),
            ),
        ] {
            let tensor = TensorData::dense_from_bytes(dtype, vec![3], bytes.clone()).unwrap();
            assert_eq!(tensor.read(0).unwrap().to_bits(), (-0f64).to_bits());
            assert!(tensor.read(1).unwrap().is_nan());
            assert!(tensor.read(2).unwrap().is_nan());
            assert_eq!(tensor.canonical_bytes().unwrap(), bytes);
        }
    }

    #[test]
    fn initialization_and_typed_payload_accounting_follow_actual_storage() {
        for dtype in [
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::I32,
            DType::U32,
            DType::Bool,
        ] {
            let representation = registry::dense(dtype);
            let mut tensor = TensorData::uninitialized(representation, vec![2]).unwrap();
            assert!(tensor.read(0).unwrap_err().contains("uninitialized"));
            tensor.write(0, scalar_from_number(dtype, 1.5)).unwrap();
            assert_eq!(tensor.read(0).unwrap(), round_to(dtype, 1.5));
            assert!(tensor.read(1).unwrap_err().contains("uninitialized"));
            assert!(tensor.canonical_bytes().is_err());
            tensor.write(1, scalar_from_number(dtype, 0.)).unwrap();
            assert_eq!(
                tensor.canonical_bytes().unwrap().len(),
                2 * dtype.bytes() as usize
            );
            assert!(tensor.read(2).is_err());
            assert!(tensor.write(2, scalar_from_number(dtype, 1.)).is_err());
            let expected = 2 * (u64::from(dtype.bytes()) + 1) + std::mem::size_of::<usize>() as u64;
            assert_eq!(
                TensorData::allocation_bytes(representation, &[2]).unwrap(),
                expected
            );
            assert_eq!(tensor.storage_bytes().unwrap(), expected);
        }
        let mut shape = Vec::with_capacity(5);
        shape.push(2);
        let mut bytes = Vec::with_capacity(20);
        bytes.extend_from_slice(&[0; 8]);
        let tensor = TensorData::dense_from_bytes(DType::F32, shape, bytes).unwrap();
        assert_eq!(
            tensor.storage_bytes().unwrap(),
            20 + 2 + 5 * std::mem::size_of::<usize>() as u64
        );
    }

    #[test]
    fn native_dense_geometry_is_validated_before_allocation() {
        assert!(TensorData::dense_from_bytes(DType::F32, vec![2], vec![0; 7]).is_err());
        assert!(TensorData::dense_from_bytes(DType::F32, vec![usize::MAX, 2], vec![]).is_err());
        let scalar =
            TensorData::dense_from_bytes(DType::F32, vec![], 1f32.to_le_bytes().to_vec()).unwrap();
        assert_eq!(scalar.read(0).unwrap(), 1.);
        let empty = TensorData::dense_from_bytes(DType::F32, vec![0], vec![]).unwrap();
        assert!(empty.canonical_bytes().unwrap().is_empty());
        assert!(empty.read(0).is_err());
    }
}
