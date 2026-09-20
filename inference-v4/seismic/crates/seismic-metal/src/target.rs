//! Effective Metal target profile: the normalized intersection of observed device/compiler
//! support, the source intrinsic registry, and realizations this backend can actually emit.

use seismic_lang::{
    intrinsics::Operation,
    sir::IntrinsicUse,
    types::{DType, Elem, Ty},
};

pub const BACKEND_IMPLEMENTATION_REVISION: &str = "seismic-metal-realizations-v2";
pub const COMPILER_PROBE_REVISION: &str = "seismic-metal-compile-probes-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatrixCombination {
    pub accumulator: DType,
    pub left: DType,
    pub right: DType,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum SignatureType {
    Integer,
    Float(DType),
    Fragment8x8(DType),
    Matrix2(DType),
    Void,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Signature {
    path: String,
    arguments: Vec<SignatureType>,
    result: SignatureType,
}

impl SignatureType {
    fn canonical(&self) -> String {
        match self {
            Self::Integer => "i32".into(),
            Self::Float(dtype) => dtype.name().into(),
            Self::Fragment8x8(dtype) => format!("fragment8x8<{}>", dtype.name()),
            Self::Matrix2(dtype) => format!("matrix2<{}>", dtype.name()),
            Self::Void => "void".into(),
        }
    }
}

impl Signature {
    fn canonical(&self) -> String {
        format!(
            "{}({})->{}",
            self.path,
            self.arguments
                .iter()
                .map(SignatureType::canonical)
                .collect::<Vec<_>>()
                .join(","),
            self.result.canonical()
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetProfile {
    signatures: Vec<Signature>,
    fingerprint: String,
}

impl TargetProfile {
    pub fn synthetic(
        max_threads: u64,
        max_threadgroup_bytes: u64,
        max_buffer_bytes: u64,
        private_budget_bytes: u64,
    ) -> Self {
        Self::from_evidence(
            "synthetic-metal-baseline",
            &[DType::F16, DType::F32],
            &[DType::F16, DType::F32],
            &[
                MatrixCombination {
                    accumulator: DType::F16,
                    left: DType::F16,
                    right: DType::F16,
                },
                MatrixCombination {
                    accumulator: DType::F32,
                    left: DType::F16,
                    right: DType::F16,
                },
                MatrixCombination {
                    accumulator: DType::F32,
                    left: DType::F16,
                    right: DType::F32,
                },
                MatrixCombination {
                    accumulator: DType::F32,
                    left: DType::F32,
                    right: DType::F16,
                },
                MatrixCombination {
                    accumulator: DType::F32,
                    left: DType::F32,
                    right: DType::F32,
                },
            ],
            max_threads,
            max_threadgroup_bytes,
            max_buffer_bytes,
            private_budget_bytes,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_evidence(
        observation: &str,
        scalar_dtypes: &[DType],
        matrix_dtypes: &[DType],
        matrix_combinations: &[MatrixCombination],
        max_threads: u64,
        max_threadgroup_bytes: u64,
        max_buffer_bytes: u64,
        private_budget_bytes: u64,
    ) -> Self {
        let mut signatures = Vec::new();
        let mut push = |path: &str, arguments: Vec<SignatureType>, result: SignatureType| {
            signatures.push(Signature {
                path: path.into(),
                arguments,
                result,
            });
        };
        push("metal.subgroup.lane_index", vec![], SignatureType::Integer);
        for &dtype in scalar_dtypes {
            let float = SignatureType::Float(dtype);
            push(
                "metal.subgroup.shuffle",
                vec![float.clone(), SignatureType::Integer],
                float.clone(),
            );
            for name in ["simd_sum", "simd_max", "simd_min"] {
                push(
                    &format!("metal.subgroup.{name}"),
                    vec![float.clone()],
                    float.clone(),
                );
            }
        }
        for &dtype in matrix_dtypes {
            let fragment = SignatureType::Fragment8x8(dtype);
            let matrix = SignatureType::Matrix2(dtype);
            push("metal.matrix.simdgroup_matrix", vec![], fragment.clone());
            for name in ["simdgroup_load", "simdgroup_load_t"] {
                push(
                    &format!("metal.matrix.{name}"),
                    vec![
                        fragment.clone(),
                        matrix.clone(),
                        SignatureType::Integer,
                        SignatureType::Integer,
                    ],
                    SignatureType::Void,
                );
            }
            push(
                "metal.matrix.simdgroup_store",
                vec![
                    fragment,
                    matrix,
                    SignatureType::Integer,
                    SignatureType::Integer,
                ],
                SignatureType::Void,
            );
        }
        for combination in matrix_combinations {
            push(
                "metal.matrix.simdgroup_multiply_accumulate",
                vec![
                    SignatureType::Fragment8x8(combination.accumulator),
                    SignatureType::Fragment8x8(combination.left),
                    SignatureType::Fragment8x8(combination.right),
                    SignatureType::Fragment8x8(combination.accumulator),
                ],
                SignatureType::Void,
            );
            push(
                "metal.matrix.matmul",
                vec![
                    SignatureType::Matrix2(combination.left),
                    SignatureType::Matrix2(combination.right),
                ],
                SignatureType::Matrix2(combination.accumulator),
            );
            push(
                "metal.matrix.matmul_add",
                vec![
                    SignatureType::Matrix2(combination.left),
                    SignatureType::Matrix2(combination.right),
                    SignatureType::Matrix2(combination.accumulator),
                ],
                SignatureType::Matrix2(combination.accumulator),
            );
        }
        signatures.sort_by_key(Signature::canonical);
        signatures.dedup();
        let signature_identity = signatures
            .iter()
            .map(Signature::canonical)
            .collect::<Vec<_>>()
            .join("|");
        let fingerprint = format!(
            "seismic-metal-target-v2;observation={observation};probe={COMPILER_PROBE_REVISION};registry={};backend={BACKEND_IMPLEMENTATION_REVISION};threads={max_threads};threadgroup={max_threadgroup_bytes};buffer={max_buffer_bytes};private={private_budget_bytes};signatures={signature_identity}",
            seismic_lang::intrinsics::REGISTRY_REVISION,
        );
        Self {
            signatures,
            fingerprint,
        }
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub fn supports_intrinsic(&self, intrinsic: &IntrinsicUse) -> Result<(), String> {
        if intrinsic.id.capability.backend != "metal" {
            return Err(format!(
                "intrinsic `{}` belongs to backend `{}`, not `metal`",
                intrinsic.id.path(),
                intrinsic.id.capability.backend
            ));
        }
        let signature = classify(intrinsic).ok_or_else(|| {
            format!(
                "invalid exact intrinsic use `{}` with arguments {:?} and result {}",
                intrinsic.id.path(),
                intrinsic.arguments,
                intrinsic.result
            )
        })?;
        self.signatures
            .contains(&signature)
            .then_some(())
            .ok_or_else(|| {
                format!(
                "effective Metal target profile does not support exact intrinsic signature `{}`",
                signature.canonical()
            )
            })
    }
}

fn scalar(ty: &Ty) -> Option<SignatureType> {
    match ty {
        Ty::Scalar(dtype) if dtype.is_float() => Some(SignatureType::Float(*dtype)),
        Ty::Scalar(DType::I32) => Some(SignatureType::Integer),
        Ty::Index(_) => Some(SignatureType::Integer),
        _ => None,
    }
}

fn fragment(ty: &Ty) -> Option<SignatureType> {
    match ty {
        Ty::Native(native)
            if native.target == "metal"
                && native.name == "simdgroup_matrix"
                && native.shape.len() == 2
                && native
                    .shape
                    .iter()
                    .all(|extent| extent.as_constant() == Some(8)) =>
        {
            match native.elem.as_ref() {
                Some(Elem::Dtype(dtype)) => Some(SignatureType::Fragment8x8(*dtype)),
                _ => None,
            }
        }
        _ => None,
    }
}

fn matrix(ty: &Ty) -> Option<SignatureType> {
    match ty {
        Ty::Tile(shape) | Ty::View(shape) => Some(shape),
        _ => None,
    }
    .and_then(|shape| {
        if shape.rank() != 2 {
            return None;
        }
        match shape.elem {
            Elem::Dtype(dtype) => Some(SignatureType::Matrix2(dtype)),
            _ => None,
        }
    })
}

fn logical_matrix(ty: &Ty) -> Option<SignatureType> {
    match ty {
        Ty::Tensor(shape) | Ty::Tile(shape) | Ty::View(shape) if shape.rank() == 2 => {
            match shape.elem {
                Elem::Dtype(dtype) => Some(SignatureType::Matrix2(dtype)),
                _ => None,
            }
        }
        _ => None,
    }
}

fn classify(intrinsic: &IntrinsicUse) -> Option<Signature> {
    let path = intrinsic.id.path();
    let (arguments, result) = match intrinsic.operation {
        Operation::LaneIndex => (vec![], scalar(&intrinsic.result)?),
        Operation::ShuffleIndex => match intrinsic.arguments.as_slice() {
            [value, index] => (
                vec![scalar(value)?, scalar(index)?],
                scalar(&intrinsic.result)?,
            ),
            _ => return None,
        },
        Operation::SimdSum | Operation::SimdMax | Operation::SimdMin => {
            let [value] = intrinsic.arguments.as_slice() else {
                return None;
            };
            (vec![scalar(value)?], scalar(&intrinsic.result)?)
        }
        Operation::Matrix => {
            if !intrinsic.arguments.is_empty() {
                return None;
            }
            (vec![], fragment(&intrinsic.result)?)
        }
        Operation::MatrixLoad | Operation::MatrixLoadTranspose | Operation::MatrixStore => {
            let [frag, tile, row, column] = intrinsic.arguments.as_slice() else {
                return None;
            };
            (
                vec![
                    fragment(frag)?,
                    matrix(tile)?,
                    scalar(row)?,
                    scalar(column)?,
                ],
                if intrinsic.result == Ty::Void {
                    SignatureType::Void
                } else {
                    return None;
                },
            )
        }
        Operation::MatrixMultiplyAccumulate => {
            let [accumulator, left, right, result] = intrinsic.arguments.as_slice() else {
                return None;
            };
            (
                vec![
                    fragment(accumulator)?,
                    fragment(left)?,
                    fragment(right)?,
                    fragment(result)?,
                ],
                if intrinsic.result == Ty::Void {
                    SignatureType::Void
                } else {
                    return None;
                },
            )
        }
        Operation::MatrixMatmul => {
            let [left, right] = intrinsic.arguments.as_slice() else {
                return None;
            };
            (
                vec![logical_matrix(left)?, logical_matrix(right)?],
                logical_matrix(&intrinsic.result)?,
            )
        }
        Operation::MatrixMatmulAdd => {
            let [left, right, accumulator] = intrinsic.arguments.as_slice() else {
                return None;
            };
            (
                vec![
                    logical_matrix(left)?,
                    logical_matrix(right)?,
                    logical_matrix(accumulator)?,
                ],
                logical_matrix(&intrinsic.result)?,
            )
        }
    };
    Some(Signature {
        path,
        arguments,
        result,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::intrinsics::{CapabilityId, IntrinsicId};
    use seismic_lang::sym::Sym;
    use seismic_lang::types::{Extent, Shaped};

    #[test]
    fn fingerprint_tracks_exact_signature_and_software_intersection() {
        let base = TargetProfile::from_evidence(
            "msl=3.2@probe;families=metal3,apple9@query",
            &[DType::F16, DType::F32],
            &[DType::F16],
            &[MatrixCombination {
                accumulator: DType::F16,
                left: DType::F16,
                right: DType::F16,
            }],
            1024,
            32 * 1024,
            1 << 30,
            128 * 1024,
        );
        let wider = TargetProfile::from_evidence(
            "msl=3.2@probe;families=metal3,apple9@query",
            &[DType::F16, DType::F32],
            &[DType::F16, DType::F32],
            &[MatrixCombination {
                accumulator: DType::F16,
                left: DType::F16,
                right: DType::F16,
            }],
            1024,
            32 * 1024,
            1 << 30,
            128 * 1024,
        );
        assert_ne!(base.fingerprint(), wider.fingerprint());
        for identity in [
            seismic_lang::intrinsics::REGISTRY_REVISION,
            BACKEND_IMPLEMENTATION_REVISION,
            COMPILER_PROBE_REVISION,
        ] {
            assert!(base.fingerprint().contains(identity));
        }
        assert!(base.fingerprint().contains("private=131072"));
        assert!(base
            .fingerprint()
            .contains("metal.subgroup.simd_sum(f32)->f32"));
    }

    fn intrinsic(operation: Operation, arguments: Vec<Ty>, result: Ty) -> IntrinsicUse {
        let (family, name) = match operation {
            Operation::SimdSum => ("subgroup", "simd_sum"),
            Operation::MatrixMatmul => ("matrix", "matmul"),
            Operation::MatrixMatmulAdd => ("matrix", "matmul_add"),
            _ => unreachable!("unsupported test intrinsic"),
        };
        IntrinsicUse {
            id: IntrinsicId {
                capability: CapabilityId::new("metal", family),
                name: name.into(),
            },
            operation,
            arguments,
            result,
        }
    }

    #[test]
    fn admission_uses_the_exact_probed_signature_set() {
        let profile = TargetProfile::from_evidence(
            "synthetic-observation",
            &[DType::F32],
            &[],
            &[],
            1024,
            32 * 1024,
            1 << 30,
            128 * 1024,
        );
        assert!(profile
            .supports_intrinsic(&intrinsic(
                Operation::SimdSum,
                vec![Ty::Scalar(DType::F32)],
                Ty::Scalar(DType::F32),
            ))
            .is_ok());
        assert!(profile
            .supports_intrinsic(&intrinsic(
                Operation::SimdSum,
                vec![Ty::Scalar(DType::BF16)],
                Ty::Scalar(DType::BF16),
            ))
            .unwrap_err()
            .contains("does not support exact intrinsic signature"));
        assert!(profile
            .supports_intrinsic(&intrinsic(
                Operation::SimdSum,
                vec![Ty::Scalar(DType::F32)],
                Ty::Scalar(DType::F16),
            ))
            .unwrap_err()
            .contains("does not support exact intrinsic signature"));
    }

    fn matrix(rows: i64, columns: i64, dtype: DType) -> Ty {
        Ty::Tensor(Shaped::new(
            vec![
                Extent::Semantic(Sym::constant(rows)),
                Extent::Semantic(Sym::constant(columns)),
            ],
            Elem::Dtype(dtype),
        ))
    }

    #[test]
    fn logical_matrix_admission_is_derived_from_exact_mma_evidence() {
        let profile = TargetProfile::from_evidence(
            "synthetic-observation",
            &[],
            &[DType::F16, DType::F32],
            &[MatrixCombination {
                accumulator: DType::F32,
                left: DType::F16,
                right: DType::F16,
            }],
            1024,
            32 * 1024,
            1 << 30,
            128 * 1024,
        );
        assert!(profile
            .supports_intrinsic(&intrinsic(
                Operation::MatrixMatmulAdd,
                vec![
                    matrix(9, 17, DType::F16),
                    matrix(17, 11, DType::F16),
                    matrix(9, 11, DType::F32),
                ],
                matrix(9, 11, DType::F32),
            ))
            .is_ok());
        assert!(profile
            .supports_intrinsic(&intrinsic(
                Operation::MatrixMatmul,
                vec![matrix(8, 8, DType::F32), matrix(8, 8, DType::F32)],
                matrix(8, 8, DType::F32),
            ))
            .unwrap_err()
            .contains("does not support exact intrinsic signature"));
    }
}
