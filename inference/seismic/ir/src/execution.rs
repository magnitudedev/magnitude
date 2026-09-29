//! One coordinated, structurally validated executable projection.

use crate::construction::ExecutableIr;
use crate::kernel::Kernel;
use crate::physical_target::PhysicalDialect;
use crate::schedule::{Launch, ParametricSchedule};
use crate::storage::LaunchLocalLayout;

/// Resources derived together from one normalized launch and its exact kernel.
/// Callers can inspect these facts but cannot assemble or replace them.
pub struct LaunchResources {
    layout: LaunchLocalLayout,
    scratch: crate::storage::LaunchScratchRequirements,
    abi: Vec<crate::storage::LaunchAbiRequirement>,
}

impl LaunchResources {
    pub fn layout(&self) -> &LaunchLocalLayout {
        &self.layout
    }
    pub fn scratch(&self) -> &crate::storage::LaunchScratchRequirements {
        &self.scratch
    }
    pub fn abi(&self) -> &[crate::storage::LaunchAbiRequirement] {
        &self.abi
    }

    fn derive<T: PhysicalDialect>(
        arena: &mut seismic_lang::expr::ExprArena,
        schedule: &ParametricSchedule<T>,
        launch: &Launch<T>,
        ordinal: usize,
        kernel: &Kernel<T>,
        policy: crate::physical_target::LocalRealizationPolicy,
        abi: &impl crate::physical_target::KernelAbiModel<T>,
    ) -> Self {
        use crate::physical_target::LocalRealization;
        use crate::storage::{LaunchLocalKind, LaunchScratchRequirements, ScratchRequirement};
        let layout = crate::storage::derive_launch_local_layout(
            arena,
            kernel.locals(),
            kernel.intrinsic_resources(),
        );
        let grid = schedule.launch_grid_envelope(arena, launch);
        let groups = arena.nat_product(&grid);
        let threads = arena.nat_product(&launch.workgroup);
        let participants = arena.nat_mul(groups, threads);
        let mut requirement = |kind, bytes| {
            let count = match policy.for_kind(kind) {
                LocalRealization::NativeDynamic | LocalRealization::NativeStatic => return None,
                LocalRealization::InvocationScratchPerWorkgroup => groups,
                LocalRealization::InvocationScratchPerParticipant => participants,
            };
            let alignment = kernel
                .locals()
                .iter()
                .filter(|local| local.kind == kind)
                .map(|local| local.alignment)
                .max()
                .unwrap_or(1);
            Some(ScratchRequirement {
                bytes: {
                    let bytes = scaled_scratch_bytes(arena, bytes, count);
                    schedule.launch_reservation(arena, ordinal, bytes)
                },
                alignment,
            })
        };
        let scratch = LaunchScratchRequirements {
            workgroup: requirement(LaunchLocalKind::Workgroup, layout.workgroup_bytes),
            participant: requirement(LaunchLocalKind::Participant, layout.participant_bytes),
            register: requirement(LaunchLocalKind::Register, layout.register_bytes),
        };
        let abi = abi
            .layout(kernel)
            .allocations
            .into_iter()
            .map(|allocation| crate::storage::LaunchAbiRequirement {
                role: allocation.role,
                bytes: arena.nat(allocation.bytes),
                alignment: allocation.alignment,
            })
            .collect();
        Self {
            layout,
            scratch,
            abi,
        }
    }

    fn retained_bytes(&self) -> usize {
        self.layout.locals.capacity() * std::mem::size_of::<crate::storage::LocalLayout>()
            + self
                .layout
                .locals
                .iter()
                .map(|local| {
                    (local.extents.capacity() + local.strides.capacity())
                        * std::mem::size_of::<seismic_lang::expr::NatExpr>()
                })
                .sum::<usize>()
            + self.abi.capacity() * std::mem::size_of::<crate::storage::LaunchAbiRequirement>()
    }
}

fn scaled_scratch_bytes(
    arena: &mut seismic_lang::expr::ExprArena,
    bytes: seismic_lang::expr::NatExpr,
    count: seismic_lang::expr::NatExpr,
) -> seismic_lang::expr::NatExpr {
    use seismic_lang::expr::{AnyExpr, NodeView};
    if matches!(arena.view(AnyExpr::Nat(bytes)), NodeView::NatConst(0)) {
        bytes
    } else {
        arena.nat_mul(bytes, count)
    }
}

/// Owned target-ready IR. Layouts cannot be separated from or outlive the IR
/// state from which they were derived; consuming transforms rebuild them.
pub struct ClosedExecutableIr<T: PhysicalDialect> {
    ir: ExecutableIr<T>,
    resources: Vec<LaunchResources>,
}

impl<T: PhysicalDialect> ClosedExecutableIr<T> {
    pub(crate) fn new(
        ir: ExecutableIr<T>,
        arena: &mut seismic_lang::expr::ExprArena,
        policy: crate::physical_target::LocalRealizationPolicy,
        abi: &impl crate::physical_target::KernelAbiModel<T>,
    ) -> Self {
        let resources = ir
            .schedule()
            .launches()
            .iter()
            .enumerate()
            .map(|(ordinal, launch)| {
                LaunchResources::derive(
                    arena,
                    ir.schedule(),
                    launch,
                    ordinal,
                    ir.kernels().kernel(launch.kernel),
                    policy,
                    abi,
                )
            })
            .collect();
        Self { ir, resources }
    }
    pub fn schedule(&self) -> &ParametricSchedule<T> {
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
    pub fn launch_resources(&self) -> &[LaunchResources] {
        &self.resources
    }
    pub fn retained_resource_bytes(&self) -> usize {
        self.resources.capacity() * std::mem::size_of::<LaunchResources>()
            + self
                .resources
                .iter()
                .map(LaunchResources::retained_bytes)
                .sum::<usize>()
    }
    pub fn into_ir(self) -> ExecutableIr<T> {
        self.ir
    }
    pub fn view(&self) -> ClosedExecutionView<'_, T> {
        ClosedExecutionView { executable: self }
    }
}

/// Borrowed executable state whose coordinated pieces were validated once at
/// the IR boundary. Downstream traversals cannot reconstruct them separately.
pub struct ClosedExecutionView<'a, T: PhysicalDialect> {
    executable: &'a ClosedExecutableIr<T>,
}

impl<T: PhysicalDialect> Copy for ClosedExecutionView<'_, T> {}
impl<T: PhysicalDialect> Clone for ClosedExecutionView<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T: PhysicalDialect> ClosedExecutionView<'a, T> {
    pub fn schedule(self) -> &'a ParametricSchedule<T> {
        self.executable.schedule()
    }

    pub fn launches(
        self,
    ) -> impl ExactSizeIterator<Item = (usize, &'a Launch<T>, &'a LaunchLocalLayout, &'a Kernel<T>)>
    {
        self.executable
            .schedule()
            .launches()
            .iter()
            .zip(self.executable.launch_resources())
            .enumerate()
            .map(|(ordinal, (launch, layout))| {
                let kernel = self.executable.kernels().kernel(launch.kernel);
                (ordinal, launch, layout.layout(), kernel)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::expr::{AnyExpr, ExprArena, NodeView, SymbolSort};
    #[test]
    fn zero_per_unit_scratch_does_not_evaluate_a_partial_launch_count() {
        let mut arena = ExprArena::default();
        let (_, start_symbol) = arena.target_constant(SymbolSort::Nat);
        let (_, end_symbol) = arena.target_constant(SymbolSort::Nat);
        let start = arena.nat_symbol(start_symbol);
        let end = arena.nat_symbol(end_symbol);
        let partial_count = arena.nat_sub(end, start);
        let zero = arena.nat(0);

        let bytes = scaled_scratch_bytes(&mut arena, zero, partial_count);
        assert!(matches!(
            arena.view(AnyExpr::Nat(bytes)),
            NodeView::NatConst(0)
        ));
        let side_conditions = arena.side_conditions(AnyExpr::Nat(bytes));
        assert!(matches!(
            arena.view(AnyExpr::Bool(side_conditions)),
            NodeView::BoolConst(true)
        ));
    }
}
