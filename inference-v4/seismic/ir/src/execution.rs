//! One coordinated, structurally validated executable projection.

use crate::construction::ExecutableIr;
use crate::identity::OwnerToken;
use crate::kernel::Kernel;
use crate::schedule::{Launch, ParametricSchedule};
use crate::storage::LaunchLocalLayout;
use crate::target::KernelDialect;

/// Layouts derived by the IR crate from one exact closed executable. The
/// owner stamp cannot be created or changed outside this crate.
pub(crate) struct ClosedLaunchLayouts {
    pub(crate) owner: OwnerToken,
    pub(crate) layouts: Vec<LaunchLocalLayout>,
}

impl ClosedLaunchLayouts {
    pub fn as_slice(&self) -> &[LaunchLocalLayout] {
        &self.layouts
    }
    pub fn retained_bytes(&self) -> usize {
        self.layouts.capacity() * std::mem::size_of::<LaunchLocalLayout>()
            + self
                .layouts
                .iter()
                .map(|layout| {
                    layout.locals.capacity() * std::mem::size_of::<crate::storage::LocalLayout>()
                        + layout
                            .locals
                            .iter()
                            .map(|local| {
                                (local.extents.capacity() + local.strides.capacity())
                                    * std::mem::size_of::<seismic_lang::expr::NatExpr>()
                            })
                            .sum::<usize>()
                })
                .sum::<usize>()
    }
}

/// Owned target-ready IR. Layouts cannot be separated from or outlive the IR
/// state from which they were derived; consuming transforms rebuild them.
pub struct ClosedExecutableIr<T: KernelDialect> {
    ir: ExecutableIr<T>,
    layouts: ClosedLaunchLayouts,
}

impl<T: KernelDialect> ClosedExecutableIr<T> {
    pub(crate) fn new(ir: ExecutableIr<T>, layouts: ClosedLaunchLayouts) -> Self {
        debug_assert_eq!(ir.owner(), layouts.owner);
        Self { ir, layouts }
    }
    pub fn schedule(&self) -> &ParametricSchedule {
        self.ir.schedule()
    }
    pub fn kernels(&self) -> &crate::kernel::KernelArena<T> {
        self.ir.kernels()
    }
    pub fn storage(&self) -> &crate::storage::GlobalAllocationTopology {
        self.ir.storage()
    }
    pub fn allocation_constraints(&self) -> &[seismic_lang::expr::BoolExpr] {
        self.ir.allocation_constraints()
    }
    pub fn local_allocations(&self) -> crate::storage::LocalAllocationTopology {
        self.ir.local_allocations()
    }
    pub fn launch_layouts(&self) -> &[LaunchLocalLayout] {
        self.layouts.as_slice()
    }
    pub fn retained_layout_bytes(&self) -> usize {
        self.layouts.retained_bytes()
    }
    pub fn into_ir(self) -> ExecutableIr<T> {
        self.ir
    }
    pub fn chunk_semantic_launches(
        self,
        arena: &mut seismic_lang::expr::ExprArena,
        chunks: impl IntoIterator<Item = (crate::schedule::LaunchId, seismic_lang::expr::NatExpr)>,
    ) -> Self {
        self.ir
            .chunk_semantic_launches(arena, chunks)
            .close_execution(arena)
    }
    pub fn view(&self) -> ClosedExecutionView<'_, T> {
        ClosedExecutionView {
            schedule: self.ir.schedule(),
            kernels: self.ir.kernels(),
            launch_layouts: self.layouts.as_slice(),
        }
    }
}

/// Borrowed executable state whose coordinated pieces were validated once at
/// the IR boundary. Downstream traversals cannot reconstruct them separately.
pub struct ClosedExecutionView<'a, T: KernelDialect> {
    schedule: &'a ParametricSchedule,
    kernels: &'a crate::kernel::KernelArena<T>,
    launch_layouts: &'a [LaunchLocalLayout],
}

impl<T: KernelDialect> Copy for ClosedExecutionView<'_, T> {}
impl<T: KernelDialect> Clone for ClosedExecutionView<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T: KernelDialect> ClosedExecutionView<'a, T> {
    pub fn schedule(self) -> &'a ParametricSchedule {
        self.schedule
    }

    pub fn launches(
        self,
    ) -> impl ExactSizeIterator<Item = (usize, &'a Launch, &'a LaunchLocalLayout, &'a Kernel<T>)>
    {
        self.schedule
            .launches()
            .iter()
            .zip(self.launch_layouts)
            .enumerate()
            .map(|(ordinal, (launch, layout))| {
                let kernel = self
                    .kernels
                    .get(launch.kernel)
                    .expect("ClosedExecutionView construction certified every kernel reference");
                (ordinal, launch, layout, kernel)
            })
    }
}
