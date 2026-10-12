use super::AttestedImport;
use crate::ImportProgramSlot;
use magnitude_kernels::{import_dense, packing_bound, repack_weight};
use seismic::{DType, Element, NativeKernel};
use std::collections::HashMap;

/// Temporary import specializations used while constructing ordered slots.
#[derive(Debug)]
pub struct ImportKernels {
    pub(super) import_dense: HashMap<(DType, DType), NativeKernel<import_dense::Entry>>,
    pub(super) repack_weight: HashMap<(Element, Element), NativeKernel<repack_weight::Entry>>,
    /// The packing bound's entry of each resident element that has one
    /// (`planning::packing_run`).
    pub(super) packing_bound: HashMap<Element, NativeKernel<packing_bound::Entry>>,
}

impl ImportKernels {
    /// The import of `slot`, where its entries were prepared: a repack
    /// carries its resident element's packing bound entry, where the
    /// element has one.
    pub(crate) fn attested(&self, slot: ImportProgramSlot) -> Option<AttestedImport> {
        match slot {
            ImportProgramSlot::Dense { source, resident } => self
                .import_dense
                .get(&(source, resident))
                .cloned()
                .map(AttestedImport::Dense),
            ImportProgramSlot::Repack { source, resident } => {
                let bound = match crate::planning::packing_run(resident) {
                    Some(_) => Some(self.packing_bound.get(&resident)?.clone()),
                    None => None,
                };
                let repack = self.repack_weight.get(&(source, resident))?.clone();
                Some(AttestedImport::Repack { repack, bound })
            }
        }
    }
}
