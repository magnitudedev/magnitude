//! Host tensors of the reference interpreter.
//!
//! Dense elements are stored as f64 but every write rounds to the tensor's
//! dtype, so results carry the same rounding a device would apply. Packed
//! tensors hold their words and coefficients and decode on read.

use crate::numeric::{bf16_round, f16_bits, f16_round, f16_to_f32};
use crate::repr;
use crate::types::DType;

#[derive(Clone, Debug)]
pub enum TensorData {
    Dense {
        dtype: DType,
        shape: Vec<usize>,
        data: Vec<f64>,
    },
    /// Physical byte planes in representation ABI order.
    Packed {
        repr: &'static repr::Repr,
        shape: Vec<usize>,
        planes: Vec<Vec<u8>>,
    },
}

impl TensorData {
    pub fn shape(&self) -> &[usize] {
        match self {
            TensorData::Dense { shape, .. } | TensorData::Packed { shape, .. } => shape,
        }
    }

    pub fn dense(dtype: DType, shape: Vec<usize>, data: Vec<f64>) -> TensorData {
        assert_eq!(data.len(), shape.iter().product::<usize>());
        TensorData::Dense { dtype, shape, data }
    }

    /// Decoded value at a flat row-major position.
    pub fn get(&self, flat: usize) -> f64 {
        match self {
            TensorData::Dense { data, .. } => data[flat],
            TensorData::Packed { repr, planes, .. } => {
                let plane_value = |plane: &repr::Plane, entry: usize| -> f32 {
                    let bytes = &planes[repr.plane_index(plane.name).unwrap()];
                    match &plane.encoding {
                        repr::PlaneEncoding::Packed {
                            bits,
                            interpretation,
                        } => interpretation.decode(repr::read_packed(bytes, entry, *bits), *bits)
                            as f32,
                        repr::PlaneEncoding::Dense(dtype) => {
                            let start = entry * dtype.bytes() as usize;
                            match dtype {
                                DType::F32 => {
                                    f32::from_le_bytes(bytes[start..start + 4].try_into().unwrap())
                                }
                                DType::F16 => f16_to_f32(u16::from_le_bytes(
                                    bytes[start..start + 2].try_into().unwrap(),
                                )),
                                DType::BF16 => f32::from_bits(
                                    u32::from(u16::from_le_bytes(
                                        bytes[start..start + 2].try_into().unwrap(),
                                    )) << 16,
                                ),
                                _ => unreachable!("nonfloating coefficient"),
                            }
                        }
                    }
                };
                let coefficient = |bias| match repr.coefficient(bias) {
                    None => 0.0,
                    Some(repr::Coefficient::Direct { plane }) => {
                        plane_value(&plane, flat / plane.group as usize)
                    }
                    Some(repr::Coefficient::Product {
                        factor,
                        coefficients,
                        field,
                        sign,
                    }) => {
                        let code = plane_value(
                            &coefficients,
                            flat / coefficients.group as usize * coefficients.fields as usize
                                + field as usize,
                        );
                        (plane_value(&factor, flat / factor.group as usize) * code) * sign as f32
                    }
                };
                let code = repr::read_packed(&planes[0], flat, repr.bits);
                (coefficient(false) as f64 * repr.decode_code(code) as f64
                    + coefficient(true) as f64) as f32 as f64
            }
        }
    }

    pub fn set(&mut self, flat: usize, value: f64) {
        match self {
            TensorData::Dense { dtype, data, .. } => data[flat] = round_to(*dtype, value),
            TensorData::Packed { .. } => panic!("cannot store into a packed tensor"),
        }
    }

    pub fn bytes(&self) -> usize {
        match self {
            TensorData::Dense { dtype, shape, .. } => {
                shape.iter().product::<usize>() * dtype.bytes() as usize
            }
            TensorData::Packed { planes, .. } => planes.iter().map(Vec::len).sum(),
        }
    }
}

/// Round an f64 to a dtype's representable value.
pub fn round_to(dtype: DType, v: f64) -> f64 {
    match dtype {
        DType::F32 => v as f32 as f64,
        DType::BF16 => bf16_round(v as f32) as f64,
        DType::F16 => f16_round(v as f32) as f64,
        DType::I32 => v as i32 as f64,
        DType::U32 => v as u32 as f64,
        DType::Bool => {
            if v != 0.0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Deterministic pseudo-random numbers for test inputs.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next() % 1_000_000) as f64 / 1_000_000.0
    }
}

impl TensorData {
    /// A dense tensor with values uniform in [-1, 1), rounded to the dtype.
    pub fn random_dense(rng: &mut Rng, dtype: DType, shape: Vec<usize>) -> TensorData {
        let n: usize = shape.iter().product();
        let data = (0..n)
            .map(|_| round_to(dtype, rng.unit() * 2.0 - 1.0))
            .collect();
        TensorData::Dense { dtype, shape, data }
    }

    /// A packed tensor with random codes and small positive scales.
    pub fn random_packed(rng: &mut Rng, rep: &'static repr::Repr, shape: Vec<usize>) -> TensorData {
        let k = *shape.last().unwrap();
        let rows: usize = shape[..shape.len() - 1].iter().product();
        assert!(
            k % rep.storage_group() as usize == 0,
            "packed rows require complete storage groups"
        );
        let count = rows * k;
        let mut planes = Vec::new();
        for plane in rep.planes() {
            let mut bytes = vec![0; plane.bytes(count as u64).unwrap() as usize];
            let entries = plane.entries(count as u64).unwrap() as usize;
            match plane.encoding {
                repr::PlaneEncoding::Packed { bits, .. } => {
                    for entry in 0..entries {
                        repr::write_packed(&mut bytes, entry, bits, rng.next() as u32);
                    }
                }
                repr::PlaneEncoding::Dense(dtype) => {
                    for entry in 0..entries {
                        let value = if plane.name == "bias" {
                            (rng.unit() - 0.5) as f32
                        } else {
                            (rng.unit() * 0.01 + 0.001) as f32
                        };
                        let start = entry * dtype.bytes() as usize;
                        match dtype {
                            DType::F32 => {
                                bytes[start..start + 4].copy_from_slice(&value.to_le_bytes())
                            }
                            DType::F16 => bytes[start..start + 2]
                                .copy_from_slice(&f16_bits(value).to_le_bytes()),
                            DType::BF16 => bytes[start..start + 2].copy_from_slice(
                                &((bf16_round(value).to_bits() >> 16) as u16).to_le_bytes(),
                            ),
                            _ => unreachable!("nonfloating coefficient"),
                        }
                    }
                }
            }
            planes.push(bytes);
        }
        TensorData::Packed {
            repr: rep,
            shape,
            planes,
        }
    }

    /// Byte images of the buffers this tensor occupies on a device, in ABI order.
    pub fn device_bytes(&self) -> Vec<Vec<u8>> {
        match self {
            TensorData::Dense { dtype, data, .. } => {
                let mut out = Vec::with_capacity(data.len() * dtype.bytes() as usize);
                for v in data {
                    match dtype {
                        DType::F32 => out.extend_from_slice(&(*v as f32).to_le_bytes()),
                        DType::BF16 => {
                            out.extend_from_slice(&((*v as f32).to_bits() >> 16).to_le_bytes()[..2])
                        }
                        DType::F16 => out.extend_from_slice(&f16_bits(*v as f32).to_le_bytes()),
                        DType::I32 => out.extend_from_slice(&(*v as i32).to_le_bytes()),
                        DType::U32 => out.extend_from_slice(&(*v as u32).to_le_bytes()),
                        DType::Bool => out.push(*v as u8),
                    }
                }
                vec![out]
            }
            TensorData::Packed { planes, .. } => planes.clone(),
        }
    }

    /// Replace a dense tensor's values from device bytes.
    pub fn load_device_bytes(&mut self, bytes: &[u8]) {
        let TensorData::Dense { dtype, data, .. } = self else {
            panic!("only dense tensors are read back")
        };
        let w = dtype.bytes() as usize;
        for (i, v) in data.iter_mut().enumerate() {
            let b = &bytes[i * w..(i + 1) * w];
            *v = match dtype {
                DType::F32 => f32::from_le_bytes(b.try_into().unwrap()) as f64,
                DType::BF16 => {
                    f32::from_bits((u16::from_le_bytes(b.try_into().unwrap()) as u32) << 16) as f64
                }
                DType::F16 => f16_to_f32(u16::from_le_bytes(b.try_into().unwrap())) as f64,
                DType::I32 => i32::from_le_bytes(b.try_into().unwrap()) as f64,
                DType::U32 => u32::from_le_bytes(b.try_into().unwrap()) as f64,
                DType::Bool => b[0] as f64,
            };
        }
    }
}
