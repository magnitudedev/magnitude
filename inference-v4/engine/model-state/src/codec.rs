use seismic::DType;
use std::{fmt, sync::Arc};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CodecIdentity(Arc<str>);

impl CodecIdentity {
    pub fn new(value: impl Into<Arc<str>>) -> Result<Self, String> {
        let value = value.into();
        if value.is_empty() {
            return Err("codec identity must not be empty".into());
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CodecIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LayerRef {
    Target(u32),
    Head(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum VectorKind {
    Key,
    Value,
}

/// A history plane of one vector kind. Affine metadata is one plane of
/// (scale, zero) pairs per group, so an attention read fetches both with one
/// load; codes never share a plane with metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PlaneName {
    Dense,
    Codes,
    Coefficients,
    Norm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    Dense {
        dtype: DType,
    },
    Affine {
        bits: u8,
        group: usize,
        scale_dtype: DType,
    },
    RotatedLloydMax {
        bits: u8,
        norm_dtype: DType,
        sign_seed: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodecSpec {
    pub key: Codec,
    pub value: Codec,
    pub key_width: usize,
    pub value_width: usize,
    pub packing: u8,
}

/// Values per (scale, zero) pair of the affine K8/V4 codec: llama.cpp's
/// q8_0/q4_0 block size. A head width must be a multiple of it.
pub const AFFINE_GROUP: usize = 32;

/// Host-selectable KV storage policy. These names are the stable public
/// options; their numerical constants are centralized here so model-family
/// adapters and executors cannot acquire private codec variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KvCodec {
    Dense,
    AffineK8V4,
    RotatedK4V4,
}

impl std::str::FromStr for KvCodec {
    type Err = String;

    /// The operator-facing names (`--kv-codec`).
    fn from_str(name: &str) -> Result<Self, String> {
        match name {
            "dense" => Ok(Self::Dense),
            "affine-k8v4" => Ok(Self::AffineK8V4),
            "rotated-k4v4" => Ok(Self::RotatedK4V4),
            _ => Err(format!(
                "unknown KV codec {name}: expected dense, affine-k8v4 or rotated-k4v4"
            )),
        }
    }
}

impl KvCodec {
    pub const fn identity(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::AffineK8V4 => "affine_k8_uniform_v4",
            Self::RotatedK4V4 => "rotated_k4_uniform_v4",
        }
    }

    /// `spec` widths are per attention head. Affine K8/V4 splits every
    /// (history row, head) vector into groups of [`AFFINE_GROUP`] values.
    pub const fn spec(self, dense_dtype: DType, key_width: usize, value_width: usize) -> CodecSpec {
        let (key, value) = match self {
            Self::Dense => (
                Codec::Dense { dtype: dense_dtype },
                Codec::Dense { dtype: dense_dtype },
            ),
            Self::AffineK8V4 => (
                Codec::Affine {
                    bits: 8,
                    group: AFFINE_GROUP,
                    scale_dtype: DType::F16,
                },
                Codec::Affine {
                    bits: 4,
                    group: AFFINE_GROUP,
                    scale_dtype: DType::F16,
                },
            ),
            Self::RotatedK4V4 => (
                Codec::RotatedLloydMax {
                    bits: 4,
                    norm_dtype: DType::F16,
                    sign_seed: 42,
                },
                Codec::Affine {
                    bits: 4,
                    group: 0,
                    scale_dtype: DType::F16,
                },
            ),
        };
        CodecSpec {
            key,
            value,
            key_width,
            value_width,
            packing: 1,
        }
    }
}

impl CodecSpec {
    pub fn dense(dtype: DType, key_width: usize, value_width: usize) -> Self {
        Self {
            key: Codec::Dense { dtype },
            value: Codec::Dense { dtype },
            key_width,
            value_width,
            packing: 1,
        }
    }

    pub fn validate(self) -> Result<Self, LayoutError> {
        if self.packing == 0 {
            return Err(LayoutError::InvalidPacking(self.packing));
        }
        planes_for(VectorKind::Key, self.key, self.key_width)?;
        planes_for(VectorKind::Value, self.value, self.value_width)?;
        Ok(self)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaneDescriptor {
    pub vector: VectorKind,
    pub name: PlaneName,
    pub dtype: DType,
    pub row_extents: Vec<usize>,
    pub row_elements: usize,
    pub row_bytes: usize,
}

/// One attention-history component: `heads` vectors of each kind per history
/// row, each encoded by `codec` (whose widths are per head). Every plane row
/// is `[heads, per-head elements]`, so a plane is `[rows, heads, elements]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComponentDescriptor {
    pub layer: LayerRef,
    pub codec: CodecSpec,
    pub heads: usize,
    planes: Vec<PlaneDescriptor>,
}

impl ComponentDescriptor {
    pub fn new(layer: LayerRef, codec: CodecSpec, heads: usize) -> Result<Self, LayoutError> {
        let codec = codec.validate()?;
        if heads == 0 {
            return Err(LayoutError::ZeroHeads);
        }
        let mut planes = planes_for(VectorKind::Key, codec.key, codec.key_width)?;
        planes.extend(planes_for(
            VectorKind::Value,
            codec.value,
            codec.value_width,
        )?);
        for plane in &mut planes {
            plane.row_extents = vec![heads, plane.row_elements];
            plane.row_elements = plane
                .row_elements
                .checked_mul(heads)
                .ok_or(LayoutError::ArithmeticOverflow("plane row elements"))?;
            plane.row_bytes = plane
                .row_bytes
                .checked_mul(heads)
                .ok_or(LayoutError::ArithmeticOverflow("plane row bytes"))?;
        }
        Ok(Self {
            layer,
            codec,
            heads,
            planes,
        })
    }

    pub fn planes(&self) -> &[PlaneDescriptor] {
        &self.planes
    }

    pub fn row_bytes(&self) -> Result<usize, LayoutError> {
        self.planes.iter().try_fold(0usize, |total, plane| {
            total
                .checked_add(plane.row_bytes)
                .ok_or(LayoutError::ArithmeticOverflow("component row bytes"))
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayoutError {
    ZeroWidth(VectorKind),
    ZeroHeads,
    InvalidDType { plane: PlaneName, dtype: DType },
    InvalidBits(u8),
    InvalidGroup { width: usize, group: usize },
    InvalidPacking(u8),
    DuplicateLayer(LayerRef),
    ArithmeticOverflow(&'static str),
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroWidth(vector) => write!(f, "{vector:?} width must be positive"),
            Self::ZeroHeads => f.write_str("an attention-history component needs a head"),
            Self::InvalidDType { plane, dtype } => {
                write!(
                    f,
                    "{plane:?} plane requires a floating dtype, got {dtype:?}"
                )
            }
            Self::InvalidBits(bits) => write!(f, "codec bit width {bits} is invalid"),
            Self::InvalidGroup { width, group } => {
                write!(f, "codec group {group} does not divide width {width}")
            }
            Self::InvalidPacking(packing) => write!(f, "packing version {packing} is invalid"),
            Self::DuplicateLayer(layer) => {
                write!(f, "duplicate attention-history component for {layer:?}")
            }
            Self::ArithmeticOverflow(field) => write!(f, "{field} arithmetic overflow"),
        }
    }
}

impl std::error::Error for LayoutError {}

fn planes_for(
    vector: VectorKind,
    codec: Codec,
    width: usize,
) -> Result<Vec<PlaneDescriptor>, LayoutError> {
    if width == 0 {
        return Err(LayoutError::ZeroWidth(vector));
    }
    match codec {
        Codec::Dense { dtype } => {
            require_float(PlaneName::Dense, dtype)?;
            Ok(vec![plane(vector, PlaneName::Dense, dtype, width)?])
        }
        Codec::Affine {
            bits,
            group,
            scale_dtype,
        } => {
            require_bits(bits)?;
            require_float(PlaneName::Coefficients, scale_dtype)?;
            let pairs = groups(width, group)?
                .checked_mul(2)
                .ok_or(LayoutError::ArithmeticOverflow("coefficient pairs"))?;
            Ok(vec![
                codes_plane(vector, width, bits)?,
                plane(vector, PlaneName::Coefficients, scale_dtype, pairs)?,
            ])
        }
        Codec::RotatedLloydMax {
            bits,
            norm_dtype,
            sign_seed: _,
        } => {
            // The engine contract currently defines the 16-centroid form only.
            if bits != 4 {
                return Err(LayoutError::InvalidBits(bits));
            }
            require_float(PlaneName::Norm, norm_dtype)?;
            Ok(vec![
                codes_plane(vector, width, bits)?,
                plane(vector, PlaneName::Norm, norm_dtype, 1)?,
            ])
        }
    }
}

fn require_bits(bits: u8) -> Result<(), LayoutError> {
    if (1..=32).contains(&bits) {
        Ok(())
    } else {
        Err(LayoutError::InvalidBits(bits))
    }
}

fn require_float(name: PlaneName, dtype: DType) -> Result<(), LayoutError> {
    if dtype.is_float() {
        Ok(())
    } else {
        Err(LayoutError::InvalidDType { plane: name, dtype })
    }
}

fn groups(width: usize, group: usize) -> Result<usize, LayoutError> {
    let divisor = if group == 0 { width } else { group };
    if !width.is_multiple_of(divisor) {
        Err(LayoutError::InvalidGroup { width, group })
    } else {
        Ok(width / divisor)
    }
}

fn codes_plane(vector: VectorKind, width: usize, bits: u8) -> Result<PlaneDescriptor, LayoutError> {
    let bit_count = width
        .checked_mul(usize::from(bits))
        .ok_or(LayoutError::ArithmeticOverflow("code row bits"))?;
    let blocks = bit_count
        .checked_add(127)
        .ok_or(LayoutError::ArithmeticOverflow("code row alignment"))?
        / 128;
    let words = blocks
        .checked_mul(4)
        .ok_or(LayoutError::ArithmeticOverflow("code row words"))?;
    plane(vector, PlaneName::Codes, DType::U32, words)
}

fn plane(
    vector: VectorKind,
    name: PlaneName,
    dtype: DType,
    row_elements: usize,
) -> Result<PlaneDescriptor, LayoutError> {
    let row_bytes = row_elements
        .checked_mul(dtype.bytes() as usize)
        .ok_or(LayoutError::ArithmeticOverflow("plane row bytes"))?;
    Ok(PlaneDescriptor {
        vector,
        name,
        dtype,
        row_extents: vec![row_elements],
        row_elements,
        row_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_and_affine_plane_arithmetic_is_exact() {
        let dense =
            ComponentDescriptor::new(LayerRef::Target(2), CodecSpec::dense(DType::F16, 128, 64), 1)
                .unwrap();
        assert_eq!(
            dense
                .planes()
                .iter()
                .map(|plane| (plane.vector, plane.name, plane.row_bytes))
                .collect::<Vec<_>>(),
            vec![
                (VectorKind::Key, PlaneName::Dense, 256),
                (VectorKind::Value, PlaneName::Dense, 128),
            ]
        );

        let affine = ComponentDescriptor::new(
            LayerRef::Head(0),
            CodecSpec {
                key: Codec::Affine {
                    bits: 8,
                    group: 32,
                    scale_dtype: DType::F16,
                },
                value: Codec::Affine {
                    bits: 4,
                    group: 0,
                    scale_dtype: DType::F16,
                },
                key_width: 96,
                value_width: 96,
                packing: 2,
            },
            1,
        )
        .unwrap();
        assert_eq!(
            affine
                .planes()
                .iter()
                .map(|plane| (plane.vector, plane.name, plane.row_bytes))
                .collect::<Vec<_>>(),
            vec![
                (VectorKind::Key, PlaneName::Codes, 96),
                (VectorKind::Key, PlaneName::Coefficients, 12),
                (VectorKind::Value, PlaneName::Codes, 48),
                (VectorKind::Value, PlaneName::Coefficients, 4),
            ]
        );
    }

    #[test]
    fn head_vectors_split_into_affine_groups() {
        // Qwen3.5-4B attention history: 4 kv heads of 256, 8 groups each.
        let affine =
            ComponentDescriptor::new(LayerRef::Target(3), KvCodec::AffineK8V4.spec(DType::BF16, 256, 256), 4)
                .unwrap();
        assert_eq!(
            affine
                .planes()
                .iter()
                .map(|plane| (plane.vector, plane.name, plane.dtype, plane.row_extents.clone(), plane.row_bytes))
                .collect::<Vec<_>>(),
            vec![
                (VectorKind::Key, PlaneName::Codes, DType::U32, vec![4, 64], 1024),
                (VectorKind::Key, PlaneName::Coefficients, DType::F16, vec![4, 16], 128),
                (VectorKind::Value, PlaneName::Codes, DType::U32, vec![4, 32], 512),
                (VectorKind::Value, PlaneName::Coefficients, DType::F16, vec![4, 16], 128),
            ]
        );
        assert_eq!(affine.row_bytes().unwrap(), 1792);
        assert!(matches!(
            ComponentDescriptor::new(LayerRef::Target(3), KvCodec::AffineK8V4.spec(DType::BF16, 48, 48), 1),
            Err(LayoutError::InvalidGroup { width: 48, group: AFFINE_GROUP })
        ));
        let dense =
            ComponentDescriptor::new(LayerRef::Target(3), KvCodec::Dense.spec(DType::BF16, 256, 256), 4)
                .unwrap();
        assert_eq!(dense.planes()[0].row_extents, [4, 256]);
        assert_eq!(dense.row_bytes().unwrap(), 4096);
        assert!(matches!(
            ComponentDescriptor::new(LayerRef::Target(3), CodecSpec::dense(DType::BF16, 8, 8), 0),
            Err(LayoutError::ZeroHeads)
        ));
    }

    #[test]
    fn named_codec_policies_bind_the_frozen_constants() {
        let affine = KvCodec::AffineK8V4.spec(DType::BF16, 96, 64);
        assert_eq!(KvCodec::AffineK8V4.identity(), "affine_k8_uniform_v4");
        assert_eq!(affine.key_width, 96);
        assert_eq!(affine.value_width, 64);
        assert_eq!(
            affine.key,
            Codec::Affine {
                bits: 8,
                group: 32,
                scale_dtype: DType::F16,
            }
        );
        assert_eq!(
            affine.value,
            Codec::Affine {
                bits: 4,
                group: 32,
                scale_dtype: DType::F16,
            }
        );

        let rotated = KvCodec::RotatedK4V4.spec(DType::F16, 128, 128);
        assert_eq!(
            rotated.key,
            Codec::RotatedLloydMax {
                bits: 4,
                norm_dtype: DType::F16,
                sign_seed: 42,
            }
        );
        assert!(matches!(rotated.value, Codec::Affine { bits: 4, .. }));

        let dense = KvCodec::Dense.spec(DType::BF16, 4, 8);
        assert_eq!(dense, CodecSpec::dense(DType::BF16, 4, 8));
    }

    #[test]
    fn code_rows_are_aligned_and_overflow_checked() {
        let descriptor = codes_plane(VectorKind::Key, 33, 4).unwrap();
        assert_eq!((descriptor.row_elements, descriptor.row_bytes), (8, 32));
        let rotated = ComponentDescriptor::new(
            LayerRef::Target(0),
            CodecSpec {
                key: Codec::RotatedLloydMax {
                    bits: 4,
                    norm_dtype: DType::F16,
                    sign_seed: 42,
                },
                value: Codec::Dense { dtype: DType::F16 },
                key_width: 33,
                value_width: 8,
                packing: 2,
            },
            1,
        )
        .unwrap();
        assert_eq!(rotated.planes()[0].name, PlaneName::Codes);
        assert_eq!(rotated.planes()[1].name, PlaneName::Norm);
        assert_eq!(rotated.planes()[1].row_bytes, 2);
        assert!(matches!(
            codes_plane(VectorKind::Key, usize::MAX, 8),
            Err(LayoutError::ArithmeticOverflow(_))
        ));
    }

    #[test]
    fn invalid_codec_shapes_are_rejected() {
        assert!(matches!(
            ComponentDescriptor::new(
                LayerRef::Target(0),
                CodecSpec {
                    key: Codec::Affine {
                        bits: 8,
                        group: 3,
                        scale_dtype: DType::F16
                    },
                    value: Codec::Dense { dtype: DType::F16 },
                    key_width: 8,
                    value_width: 8,
                    packing: 2,
                },
                1,
            ),
            Err(LayoutError::InvalidGroup { .. })
        ));
    }
}
