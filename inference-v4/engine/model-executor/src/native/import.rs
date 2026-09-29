use magnitude_model_kernels::{import_dense, repack_weight};
use seismic::{DType, Element, NativeKernel};
use std::collections::HashMap;

/// Temporary import specializations used while constructing ordered slots.
#[derive(Debug)]
pub struct ImportKernels {
    pub(super) import_dense: HashMap<(DType, DType), NativeKernel<import_dense::Entry>>,
    pub(super) repack_weight: HashMap<(Element, Element), NativeKernel<repack_weight::Entry>>,
}
