//! What the kernel tests share.

use seismic_lang::checked::CheckedModule;

pub mod bound;
pub mod recurrent_bound;

/// The kernels' checked module: the one the build checked and embedded
/// (`magnitude_kernels::module`), every kernel source over the standard
/// library, as the engine prepares and interprets its entries. A test reads
/// the kernels through it, never through a source list of its own, so a
/// file a kernel comes to depend on needs no test to name it.
#[allow(dead_code)]
pub fn kernels() -> &'static CheckedModule {
    magnitude_kernels::module().expect("the kernel bundle decodes").checked()
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

/// The ordinal of `E`'s Metal launch `launch`, its kernel `<entry>_<launch>`.
#[allow(dead_code)]
pub fn metal_launch<E: seismic::Entry>(launch: &str) -> usize {
    metal_launch_of::<E>(launch).unwrap_or_else(|| panic!("`{}` has no Metal launch `{launch}`", E::NAME))
}

/// The declaration index of `E`'s Metal launch `launch`, where it has one.
pub fn metal_launch_of<E: seismic::Entry>(launch: &str) -> Option<usize> {
    let native = seismic::generated::native_implementation_for_backend::<E>(seismic::BackendName::Metal)
        .expect("the kernel bundle decodes")
        .unwrap_or_else(|| panic!("`{}` has a Metal implementation", E::NAME));
    let kernel = format!("{}_{launch}", E::NAME);
    native.launches.iter().position(|candidate| candidate.kernel == kernel)
}
