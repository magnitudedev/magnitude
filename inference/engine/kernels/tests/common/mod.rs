//! What the kernel tests share.

use seismic_lang::checked::{CheckedModule, SourceSet};

pub mod bound;
pub mod recurrent_bound;

/// `sources`, held in memory, checked with each native implementation's
/// source read from the crate's `kernels/`, where the declarations name
/// them.
pub fn check(sources: SourceSet) -> CheckedModule {
    let manifest = std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    let kernels = std::path::Path::new(&manifest).join("kernels");
    seismic_lang::source::check_beside(sources, &kernels).expect("the kernel sources check")
}

/// The granule pool of `device`'s device store, its granules each the
/// largest of `layouts`' slabs, rounded up to a power of two of at least
/// 1 MiB so that the slab stores one test binds beside each other, of
/// differing small layouts, are granules of the one pool a device holds per
/// store.
#[allow(dead_code)]
pub fn slab_pool(device: &seismic::Device, layouts: &[seismic::SlabLayout]) -> seismic::Pool {
    let slab = layouts.iter().map(|layout| layout.slab_bytes).max().expect("a pool holds a layout");
    let granule = slab.next_power_of_two().max(1 << 20);
    let contract = seismic::Contract::DeviceOnly;
    device.pool(contract, granule, 2 * granule, seismic::PoolEntries::one(device.entry(contract))).unwrap()
}

/// A slab store of `regions`, `rows_per_slab` rows a slab over
/// `logical_rows`, in the pool of `device`'s device store sized for it.
#[allow(dead_code)]
pub fn slab_tensor(
    device: &seismic::Device,
    rows_per_slab: u64,
    logical_rows: u64,
    regions: &[seismic::SlabRegion],
) -> seismic::SlabTensor {
    let layout = seismic::SlabLayout::for_regions(rows_per_slab, logical_rows, regions).unwrap();
    seismic::SlabTensor::in_pool(&slab_pool(device, &[layout]), rows_per_slab, logical_rows, regions).unwrap()
}
