//! Host storage used by the semantic oracle.

use crate::ids::RepresentationId;
use crate::registry::{self, DecodeStep, PlaneEncoding, RepresentationKind};
use crate::types::DType;

#[derive(Clone, Debug)]
pub enum TensorData {
    Dense {
        representation: RepresentationId,
        shape: Vec<usize>,
        data: Vec<f64>,
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
    pub fn dense(dtype: DType, shape: Vec<usize>, values: Vec<f64>) -> Self {
        assert_eq!(values.len(), shape.iter().product::<usize>());
        let values = values
            .into_iter()
            .map(|value| round_to(dtype, value))
            .collect();
        Self::Dense {
            representation: registry::dense(dtype),
            initialized: vec![true; shape.iter().product()],
            shape,
            data: values,
        }
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
            RepresentationKind::Dense(_) => {
                let count = shape.iter().product();
                Ok(Self::Dense {
                    representation,
                    shape,
                    data: vec![0.0; count],
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

    pub(super) fn read(&self, flat: usize) -> Result<f64, String> {
        match self {
            Self::Dense {
                data, initialized, ..
            } => match (data.get(flat), initialized.get(flat)) {
                (Some(value), Some(true)) => Ok(*value),
                (Some(_), Some(false)) => Err("read of uninitialized tensor element".to_owned()),
                _ => Err("read outside tensor bounds".to_owned()),
            },
            Self::Encoded {
                representation,
                shape,
                bytes,
            } => decode(*representation, shape, bytes, flat),
        }
    }

    pub(super) fn write(&mut self, flat: usize, value: (DType, f64)) -> Result<(), String> {
        match self {
            Self::Dense {
                representation,
                data,
                initialized,
                ..
            } => {
                let RepresentationKind::Dense(dtype) =
                    &registry::representation_info(*representation).kind
                else {
                    unreachable!("dense oracle storage has a non-dense representation")
                };
                let slot = data.get_mut(flat).ok_or("write outside tensor bounds")?;
                *slot = round_to(*dtype, value.1);
                initialized[flat] = true;
                Ok(())
            }
            Self::Encoded { .. } => Err("read-only representation cannot be written".to_owned()),
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

fn encoded_bytes(representation: RepresentationId, shape: &[usize]) -> Result<usize, String> {
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
) -> Result<f64, String> {
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
    let mut temporaries = vec![0.0; recipe.temporary_count()];
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
            } => f64::from(interpretation.decode(temporaries[recipe.ordinal(*raw)] as u32, *bits)),
            DecodeStep::DecodeFloatCode { raw, format, .. } => {
                f64::from(format.decode(temporaries[recipe.ordinal(*raw)] as u32))
            }
            DecodeStep::ConvertToF32 { from, .. } => {
                temporaries[recipe.ordinal(*from)] as f32 as f64
            }
            DecodeStep::Multiply { left, right, .. } => {
                (temporaries[recipe.ordinal(*left)] as f32
                    * temporaries[recipe.ordinal(*right)] as f32) as f64
            }
            DecodeStep::Negate { from, .. } => -temporaries[recipe.ordinal(*from)],
            DecodeStep::MultiplyAdd {
                factor,
                multiplicand,
                addend,
                ..
            } => (temporaries[recipe.ordinal(*factor)] as f32).mul_add(
                temporaries[recipe.ordinal(*multiplicand)] as f32,
                temporaries[recipe.ordinal(*addend)] as f32,
            ) as f64,
            DecodeStep::Cast { from, to, .. } => round_to(*to, temporaries[recipe.ordinal(*from)]),
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
) -> Result<f64, String> {
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
        PlaneEncoding::Packed { .. } | PlaneEncoding::FloatCode { .. } => f64::from(read_bits(
            plane_bytes,
            entry as usize * schema.entry_bits as usize,
            schema.entry_bits,
        )),
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

fn read_dense(dtype: DType, bytes: &[u8]) -> f64 {
    match dtype {
        DType::F32 => f32::from_le_bytes(bytes.try_into().unwrap()) as f64,
        DType::F16 => f16_to_f32(u16::from_le_bytes(bytes.try_into().unwrap())) as f64,
        DType::BF16 => {
            f32::from_bits(u32::from(u16::from_le_bytes(bytes.try_into().unwrap())) << 16) as f64
        }
        DType::I32 => i32::from_le_bytes(bytes.try_into().unwrap()) as f64,
        DType::U32 => u32::from_le_bytes(bytes.try_into().unwrap()) as f64,
        DType::Bool => f64::from(u8::from(bytes[0] != 0)),
    }
}

pub fn round_to(dtype: DType, value: f64) -> f64 {
    match dtype {
        DType::F32 => value as f32 as f64,
        DType::F16 => f16_to_f32(f16_bits(value as f32)) as f64,
        DType::BF16 => bf16_round(value as f32) as f64,
        DType::I32 => value as i32 as f64,
        DType::U32 => value as u32 as f64,
        DType::Bool => f64::from(u8::from(value != 0.0)),
    }
}

pub(super) fn bf16_round(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    f32::from_bits(rounded & 0xffff_0000)
}

pub(super) fn f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    if exponent <= 0 {
        if exponent < -10 {
            return sign;
        }
        let significand = mantissa | 0x80_0000;
        let shift = (14 - exponent) as u32;
        let half = 1u32 << (shift - 1);
        return sign | ((significand + half - 1 + ((significand >> shift) & 1)) >> shift) as u16;
    }
    if exponent >= 31 {
        return sign | if mantissa == 0 { 0x7c00 } else { 0x7e00 };
    }
    let rounded = mantissa + 0xfff + ((mantissa >> 13) & 1);
    let mut result = sign | ((exponent as u16) << 10) | ((rounded >> 13) as u16);
    if rounded & 0x80_0000 != 0 {
        result = sign | (((exponent + 1) as u16) << 10);
    }
    result
}

pub(super) fn f16_to_f32(value: u16) -> f32 {
    let sign = (u32::from(value & 0x8000)) << 16;
    let exponent = (value >> 10) & 0x1f;
    let mantissa = u32::from(value & 0x03ff);
    let bits = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let shift = mantissa.leading_zeros() - 21;
            sign | ((127 - 15 - shift + 1) << 23) | ((mantissa << shift) & 0x7f_ffff)
        }
        31 => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((u32::from(exponent) + 127 - 15) << 23) | (mantissa << 13),
    };
    f32::from_bits(bits)
}
