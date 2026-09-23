//! Deterministic argument generation (design A10 §2.2.2 "Generation semantics").
//!
//! The corpus never computes a tensor's storage size: the caller supplies it
//! from the public API. Packed plane layouts are read from the registry, the
//! single owner of packed layouts; literal quantization is the language's own
//! (`reference_math::{float_literal, integer_literal}`).
use crate::scenario::{element_count, ArgumentSpec, Fill, Literal, ScalarDtype};
use seismic_lang::reference_math::{float_literal, integer_literal};
use seismic_lang::registry::{self, RepresentationKind};
use seismic_lang::types::DType;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GeneratedArgument {
    Tensor {
        element: String,
        shape: Vec<u64>,
        bytes: Vec<u8>,
    },
    Scalar {
        dtype: ScalarDtype,
        bits: u64,
    },
    Index(u64),
    Range {
        start: u64,
        end: u64,
    },
}

/// Generates one argument. `storage_bytes(element, shape)` is asked only for
/// tensors and must answer from the public API (`Tensor::zeros(..).storage_bytes()`).
/// The same spec and storage size give identical bytes on every machine.
pub fn generate<E>(
    spec: &ArgumentSpec,
    storage_bytes: impl FnOnce(&str, &[u64]) -> Result<u64, E>,
) -> Result<GeneratedArgument, E> {
    Ok(match spec {
        ArgumentSpec::Tensor {
            element,
            shape,
            fill,
        } => {
            let storage = usize::try_from(storage_bytes(element, shape)?)
                .expect("a generated tensor fits in host memory");
            GeneratedArgument::Tensor {
                element: element.clone(),
                shape: shape.clone(),
                bytes: tensor_bytes(element, shape, fill, storage),
            }
        }
        ArgumentSpec::Scalar { dtype, value } => GeneratedArgument::Scalar {
            dtype: *dtype,
            bits: literal_bits(dtype.dtype(), *value).expect("validated by the scenario parser"),
        },
        ArgumentSpec::Index(value) => GeneratedArgument::Index(*value),
        ArgumentSpec::Range { start, end } => GeneratedArgument::Range {
            start: *start,
            end: *end,
        },
    })
}

/// The bit pattern of `literal` as one `dtype` element, or why it has none.
pub fn literal_bits(dtype: DType, literal: Literal) -> Result<u64, String> {
    let width = dtype.bytes() * 8;
    let bits = match (dtype, literal) {
        (DType::Bool, Literal::Bool(value)) => u32::from(value),
        (DType::Bool, Literal::Bits(bits @ (0 | 1))) => bits as u32,
        (_, Literal::Bits(bits)) if dtype != DType::Bool && bits >> width == 0 => bits as u32,
        (dtype, Literal::Decimal(value)) if dtype.is_float() => {
            float_literal(dtype, value).bits()
        }
        (dtype, Literal::Integer(value)) if dtype.is_float() => {
            integer_literal(dtype, value).bits()
        }
        (DType::I32, Literal::Integer(value)) if i32::try_from(value).is_ok() => {
            integer_literal(DType::I32, value).bits()
        }
        (DType::U32, Literal::Integer(value)) if u32::try_from(value).is_ok() => {
            integer_literal(DType::U32, value).bits()
        }
        _ => {
            return Err(format!(
                "{literal:?} is not a {} element",
                dtype.name()
            ))
        }
    };
    Ok(u64::from(bits))
}

fn tensor_bytes(element: &str, shape: &[u64], fill: &Fill, storage: usize) -> Vec<u8> {
    let representation = registry::representation(element).expect("validated by the parser");
    match (fill, &registry::representation_info(representation).kind) {
        (Fill::Zero, _) => vec![0; storage],
        (Fill::Bits { seed }, _) => SplitMix64(*seed).bytes(storage),
        (Fill::Uniform { seed }, RepresentationKind::External(_)) => {
            SplitMix64(*seed).bytes(storage)
        }
        (Fill::Uniform { seed }, RepresentationKind::Packed(layout)) => {
            let mut random = SplitMix64(*seed);
            let mut bytes = vec![0; storage];
            for packet in bytes.chunks_exact_mut(layout.packet_size as usize) {
                for plane in &layout.planes {
                    let start = plane.offset as usize;
                    let region = &mut packet[start..start + plane.bytes_per_group as usize];
                    if plane.storage_dtype.is_float() {
                        let width = plane.storage_dtype.bytes() as usize;
                        for value in region.chunks_exact_mut(width) {
                            let scaled = random.grid() / 32.0;
                            let bits = float_literal(plane.storage_dtype, scaled).bits();
                            value.copy_from_slice(&bits.to_le_bytes()[..width]);
                        }
                    } else {
                        region.copy_from_slice(&random.bytes(region.len()));
                    }
                }
            }
            bytes
        }
        (Fill::Uniform { seed }, RepresentationKind::Dense(dtype)) => {
            let mut random = SplitMix64(*seed);
            dense_elements(*dtype, shape, |_| random.element_bits(*dtype))
        }
        (Fill::Sequence, RepresentationKind::Dense(dtype)) => {
            dense_elements(*dtype, shape, |k| sequence_bits(*dtype, k))
        }
        (Fill::Values(values), RepresentationKind::Dense(dtype)) => {
            dense_elements(*dtype, shape, |k| {
                literal_bits(*dtype, values[k as usize]).expect("validated by the parser")
            })
        }
        (Fill::Sequence | Fill::Values(_), _) => {
            unreachable!("the parser rejects `seq` and literal fills of packed elements")
        }
    }
}

/// Row-major little-endian elements; `bits(k)` is element `k`'s bit pattern.
fn dense_elements(dtype: DType, shape: &[u64], mut bits: impl FnMut(u64) -> u64) -> Vec<u8> {
    let count = element_count(shape).expect("validated by the parser");
    let width = dtype.bytes() as usize;
    (0..count)
        .flat_map(|k| bits(k).to_le_bytes().into_iter().take(width))
        .collect()
}

/// Element `k` of `seq`: `k` rounded to nearest-even in the dtype; `k % 2 == 1` for bool.
fn sequence_bits(dtype: DType, k: u64) -> u64 {
    if dtype == DType::Bool {
        return k % 2;
    }
    u64::from(integer_literal(dtype, i128::from(k)).bits())
}

/// SplitMix64 (Steele, Lea, Flood 2014): the only random source of the corpus.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn bytes(&mut self, len: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(len.next_multiple_of(8));
        while bytes.len() < len {
            bytes.extend_from_slice(&self.next().to_le_bytes());
        }
        bytes.truncate(len);
        bytes
    }

    /// A value in [-4, 4) on a 1/64 grid; exact in every float dtype.
    fn grid(&mut self) -> f64 {
        ((self.next() % 512) as f64 - 256.0) / 64.0
    }

    fn element_bits(&mut self, dtype: DType) -> u64 {
        match dtype {
            DType::F32 | DType::F16 | DType::BF16 => {
                u64::from(float_literal(dtype, self.grid()).bits())
            }
            DType::I32 => u64::from((((self.next() % 32) as i32) - 16) as u32),
            DType::U32 => self.next() % 32,
            DType::Bool => self.next() >> 63,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario::parse_arguments;

    fn generated(args: &str, storage: u64) -> Vec<GeneratedArgument> {
        parse_arguments(args)
            .unwrap()
            .iter()
            .map(|spec| generate(spec, |_, _| Ok::<_, ()>(storage)).unwrap())
            .collect()
    }

    fn tensor(argument: &GeneratedArgument) -> &[u8] {
        let GeneratedArgument::Tensor { bytes, .. } = argument else {
            panic!("tensor");
        };
        bytes
    }

    #[test]
    fn dense_fills_follow_the_grammar() {
        let args = generated("f32[3]=seq ; i32[4]=0,2,-1,3 ; bool[3]=seq ; f16[1]=0x3c00", 0);
        assert_eq!(tensor(&args[0]), [0f32, 1.0, 2.0].map(f32::to_le_bytes).concat());
        assert_eq!(tensor(&args[1]), [0i32, 2, -1, 3].map(i32::to_le_bytes).concat());
        assert_eq!(tensor(&args[2]), [0, 1, 0]);
        assert_eq!(tensor(&args[3]), [0x00, 0x3c]);
    }

    #[test]
    fn scalars_use_language_literal_quantization() {
        let args = generated("f32:0x3a000800 ; f16:1.5 ; bf16:-0 ; i32:-7 ; bool:true", 0);
        let bits: Vec<_> = args
            .iter()
            .map(|a| match a {
                GeneratedArgument::Scalar { bits, .. } => *bits,
                _ => panic!("scalar"),
            })
            .collect();
        assert_eq!(bits, [0x3a00_0800, 0x3e00, 0x8000, (-7i32) as u32 as u64, 1]);
        assert!(literal_bits(DType::I32, Literal::Integer(1 << 40)).is_err());
        assert!(literal_bits(DType::U32, Literal::Decimal(1.5)).is_err());
    }

    #[test]
    fn random_fills_are_deterministic_and_on_the_grid() {
        let a = generated("f32[64]=rand(1) ; q4k[1,256]=rand(4) ; f32[8]=bits(3)", 256);
        let b = generated("f32[64]=rand(1) ; q4k[1,256]=rand(4) ; f32[8]=bits(3)", 256);
        assert_eq!(a, b);
        for value in tensor(&a[0]).chunks_exact(4) {
            let value = f32::from_le_bytes(value.try_into().unwrap());
            assert!((-4.0..4.0).contains(&value) && (value * 64.0).fract() == 0.0);
        }
        assert_eq!(tensor(&a[2]).len(), 256);
    }
}
