//! Backend intrinsic tables. An intrinsic is the floor: a backend operation
//! with no body in the language, defined here with its signature, its
//! ownership where it has fragment operands, and (later) its printer.

use crate::sym::Sym;
use crate::types::{DType, Elem, Shaped, Ty};

#[derive(Clone, Debug)]
pub struct Intrinsic {
    pub name: &'static str,
    pub params: Vec<IntrinsicParam>,
    pub result: IntrinsicResult,
    /// Whether this operation can write tensor backing memory. Local tiles and
    /// fragments are value storage and do not alias tensor backing.
    pub writes_tensor_memory: bool,
    /// Mutable value-storage parameters (zero-based); all other operands are read-only.
    pub writes_arguments: &'static [usize],
}

#[derive(Clone, Debug)]
pub enum IntrinsicParam {
    /// Any scalar of a float dtype; all `FloatScalar` params of one call share a dtype.
    FloatScalar,
    /// A dtype name, e.g. `f32`.
    DTypeName,
    /// An 8x8 fragment of any dtype.
    Frag8x8,
    /// A tile (or tile view) of rank 2.
    Tile2,
    /// A symbolic integer.
    Int,
}

#[derive(Clone, Debug)]
pub enum IntrinsicResult {
    Void,
    /// Same dtype as the `FloatScalar` params.
    FloatScalar,
    /// An 8x8 fragment of the dtype named by the `DTypeName` param.
    Frag8x8OfNamedDtype,
}

pub fn table(backend: &str) -> Option<Vec<Intrinsic>> {
    use IntrinsicParam::*;
    match backend {
        "metal" => Some(vec![
            Intrinsic { writes_tensor_memory: false, writes_arguments: &[], name: "simd_sum", params: vec![FloatScalar], result: IntrinsicResult::FloatScalar },
            Intrinsic { writes_tensor_memory: false, writes_arguments: &[], name: "simd_max", params: vec![FloatScalar], result: IntrinsicResult::FloatScalar },
            Intrinsic { writes_tensor_memory: false, writes_arguments: &[], name: "simd_min", params: vec![FloatScalar], result: IntrinsicResult::FloatScalar },
            Intrinsic { writes_tensor_memory: false, writes_arguments: &[], name: "simdgroup_matrix", params: vec![DTypeName], result: IntrinsicResult::Frag8x8OfNamedDtype },
            Intrinsic { writes_tensor_memory: false, writes_arguments: &[0], name: "simdgroup_load", params: vec![Frag8x8, Tile2, Int, Int], result: IntrinsicResult::Void },
            Intrinsic { writes_tensor_memory: false, writes_arguments: &[0], name: "simdgroup_load_t", params: vec![Frag8x8, Tile2, Int, Int], result: IntrinsicResult::Void },
            Intrinsic { writes_tensor_memory: false, writes_arguments: &[1], name: "simdgroup_store", params: vec![Frag8x8, Tile2, Int, Int], result: IntrinsicResult::Void },
            Intrinsic { writes_tensor_memory: false, writes_arguments: &[0], name: "simdgroup_multiply_accumulate", params: vec![Frag8x8, Frag8x8, Frag8x8, Frag8x8], result: IntrinsicResult::Void },
        ]),
        "cpu" => Some(vec![]),
        "cuda" => Some(vec![]),
        "vulkan" => Some(vec![]),
        _ => None,
    }
}

pub fn frag8x8(dtype: DType) -> Ty {
    Ty::Frag(Shaped::new(vec![Sym::constant(8), Sym::constant(8)], Elem::Dtype(dtype)))
}
