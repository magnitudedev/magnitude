//! Launch-local storage: allocations declared by one kernel and their layout.
use super::*;

/// A launch-local allocation declared by one kernel. Not convertible to a
/// global buffer.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LaunchLocalId<R: Representation> {
    owner: OwnerToken,
    kernel: u32,
    index: u32,
    repr: PhantomData<R>,
}

impl<R: Representation> Clone for LaunchLocalId<R> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<R: Representation> Copy for LaunchLocalId<R> {}
impl<R: Representation> fmt::Debug for LaunchLocalId<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?}/local<{}>#{}.{}",
            self.owner,
            R::NAME,
            self.kernel,
            self.index
        )
    }
}

impl<R: Representation> LaunchLocalId<R> {
    pub(crate) fn new(owner: OwnerToken, kernel: u32, index: u32) -> Self {
        Self {
            owner,
            kernel,
            index,
            repr: PhantomData,
        }
    }
    pub(crate) fn kernel(self) -> u32 {
        self.kernel
    }
    pub fn index(self) -> u32 {
        self.index
    }
    pub(crate) fn owner(self) -> OwnerToken {
        self.owner
    }
}

/// Kinds of launch-local storage (§8.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LaunchLocalKind {
    Workgroup,
    Participant,
    Register,
}

/// One launch-local allocation's facts.
#[derive(Clone, Debug)]
pub struct LocalAllocation {
    pub kind: LaunchLocalKind,
    pub representation: RepresentationId,
    pub extents: Vec<NatExpr>,
    pub alignment: u64,
}

#[derive(Clone, Debug)]
pub struct LocalLayout {
    pub kind: LaunchLocalKind,
    pub representation: RepresentationId,
    pub offset: NatExpr,
    pub extents: Vec<NatExpr>,
    pub strides: Vec<NatExpr>,
    pub bytes: NatExpr,
    pub alignment: u64,
}

#[derive(Clone, Debug)]
pub struct LaunchLocalLayout {
    pub locals: Vec<LocalLayout>,
    pub workgroup_bytes: NatExpr,
    pub participant_bytes: NatExpr,
    pub register_bytes: NatExpr,
}

/// One compiler-owned physical scratch allocation required to realize a
/// launch-local address space on the selected target profile.
#[derive(Clone, Debug)]
pub struct ScratchRequirement {
    pub bytes: NatExpr,
    pub alignment: u64,
}

#[derive(Clone, Debug, Default)]
pub struct LaunchScratchRequirements {
    pub workgroup: Option<ScratchRequirement>,
    pub participant: Option<ScratchRequirement>,
    pub register: Option<ScratchRequirement>,
}

#[derive(Clone, Debug)]
pub struct LaunchAbiRequirement {
    pub role: crate::target::KernelAbiAllocationRole,
    pub bytes: NatExpr,
    pub alignment: u64,
}

pub fn derive_launch_local_layout(
    arena: &mut ExprArena,
    locals: &[LocalAllocation],
    intrinsic_resources: &[crate::kernel::ops::IntrinsicResources],
) -> LaunchLocalLayout {
    let zero = arena.nat(0);
    let mut cursors = [zero, zero, zero];
    let mut layouts = Vec::with_capacity(locals.len());
    for local in locals {
        assert!(
            local.alignment.is_power_of_two(),
            "launch-local alignment must be a nonzero power of two"
        );
        let class = match local.kind {
            LaunchLocalKind::Workgroup => 0,
            LaunchLocalKind::Participant => 1,
            LaunchLocalKind::Register => 2,
        };
        let alignment = arena.nat(local.alignment);
        let groups = arena.nat_ceil_div(cursors[class], alignment);
        let offset = arena.nat_mul(groups, alignment);
        let bytes = tensor_bytes(arena, local.representation, &local.extents);
        cursors[class] = arena.nat_add(offset, bytes);
        layouts.push(LocalLayout {
            kind: local.kind,
            representation: local.representation,
            offset,
            extents: local.extents.clone(),
            strides: dense_strides(arena, local.representation, &local.extents),
            bytes,
            alignment: local.alignment,
        });
    }
    for resources in intrinsic_resources {
        for (class, bytes) in [
            resources.workgroup_bytes,
            resources.participant_bytes,
            resources.register_bytes,
        ]
        .into_iter()
        .enumerate()
        {
            if let Some(bytes) = bytes {
                cursors[class] = arena.nat_add(cursors[class], bytes);
            }
        }
    }
    LaunchLocalLayout {
        locals: layouts,
        workgroup_bytes: cursors[0],
        participant_bytes: cursors[1],
        register_bytes: cursors[2],
    }
}
