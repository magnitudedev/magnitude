//! The CPU native ABI. CPU native implementations are Rust compiled into the
//! binary; `seismic-build` wraps each entry's source in a typed context over
//! [`CpuInvocation`] and registers one function per launch and admissible
//! tuning configuration.

/// One launch function: the invocation, the work item's group coordinate,
/// and the item's private `shared_bytes` buffer.
pub type CpuKernelFn = fn(&CpuInvocation<'_>, [u64; 3], &mut [u8]);

/// The monomorphized variants of one launch, keyed by tuning parameter
/// values in declaration order.
pub struct CpuLaunchVariants {
    pub kernel: &'static str,
    pub variants: &'static [(&'static [u64], CpuKernelFn)],
}

/// Every launch of one CPU native implementation, in declaration order.
pub struct CpuNativeKernels {
    pub launches: &'static [CpuLaunchVariants],
}

/// The ABI of one CPU native call, shared by all its work items. Buffers are
/// in ABI order (tensor parameters, tensor results, scratch). Words have the
/// same layout as the GPU argument words.
pub struct CpuInvocation<'a> {
    pub(crate) buffers: &'a [*mut u8],
    /// Registry name of each buffer's representation; scratch is `"bytes"`.
    pub(crate) representations: &'a [&'static str],
    pub(crate) words: &'a [u64],
    pub(crate) scalar_results: *mut u64,
    pub(crate) groups: [u64; 3],
    pub(crate) threads: [u64; 3],
}

// Work items run concurrently and write disjoint regions through the raw
// pointers; the pointed-to storage outlives the synchronous launch.
unsafe impl Sync for CpuInvocation<'_> {}

impl CpuInvocation<'_> {
    /// Base address of buffer `index` (already offset to the tensor view).
    pub fn buffer(&self, index: usize) -> *mut u8 {
        self.buffers[index]
    }
    pub fn representation(&self, index: usize) -> &'static str {
        self.representations[index]
    }
    pub fn word(&self, index: usize) -> u64 {
        self.words[index]
    }
    /// Slot `index` of the scalar results. Only one work item may write a
    /// given slot.
    pub fn scalar_result(&self, index: usize) -> *mut u64 {
        // SAFETY: the runtime sizes the scalar storage to every slot of the
        // checked schema, and generated contexts index only those slots.
        unsafe { self.scalar_results.add(index) }
    }
    /// Groups of the current launch on each axis.
    pub fn groups(&self) -> [u64; 3] {
        self.groups
    }
    /// Declared participants of one group on each axis.
    pub fn threads(&self) -> [u64; 3] {
        self.threads
    }
}

/// A tensor argument or result as a CPU native kernel sees it: its base
/// address (already offset to the view), extents and strides from the ABI
/// words, and representation name.
#[derive(Clone, Copy, Debug)]
pub struct CpuTensor<const RANK: usize> {
    pub pointer: *mut u8,
    pub extents: [u64; RANK],
    pub strides: [u64; RANK],
    pub representation: &'static str,
}

impl CpuNativeKernels {
    /// The function of launch `launch` for a tuning configuration.
    pub(crate) fn function(&self, launch: usize, configuration: &[u64]) -> Option<CpuKernelFn> {
        self.launches
            .get(launch)?
            .variants
            .iter()
            .find(|(values, _)| *values == configuration)
            .map(|(_, function)| *function)
    }
}
