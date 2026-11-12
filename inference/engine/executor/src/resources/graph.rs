//! Bounded Seismic graph storage owned by the executor resource plan.

use crate::{CapacityError, GraphSlots, InvariantError, ResourceDomainId, ResourceKind, planning::class_bytes};
use seismic::{
    NativeExecutionArena, NativeGraphFamily, NativeGraphFamilyOutputSlot, NativeGraphFamilySlot,
    NativeGraphOutputs, NativeGraphPlan, Tensor, WorkflowTensor,
};
use std::{cell::RefCell, rc::Rc};

/// One family's graph storage. Activations cover the launches of the family
/// in flight at once; each holds its upload regions and binds the device's
/// one workspace arena. Output slots outlive their launches (a request
/// retains its last published features), so their count follows the live
/// requests: they grow one slot at a time under a heap claim. Both start at
/// the floor of a prompt group of one place and grow to the class a group is
/// granted (`hold`); what idles beyond the held class is released as
/// surplus, and the held class itself down to the floor in Reclaim.
pub struct NativeGraphPool {
    domain: ResourceDomainId,
    family: NativeGraphFamily,
    arena: NativeExecutionArena,
    upload_regions: Vec<u64>,
    /// Upload regions of one activation.
    activation_bytes: u64,
    output_bytes: u64,
    /// Who reads its outputs ([`Outputs`]).
    outputs: Outputs,
    /// The slots of a group of one place, never released.
    floor: GraphSlots,
    /// The slots of the class held now: the floor or more.
    held: GraphSlots,
    /// Activations and output slots allocated now, lent or free.
    allocated: GraphSlots,
    workspace: FreeSlots<NativeGraphFamilySlot>,
    output: FreeSlots<NativeGraphFamilyOutputSlot>,
}

/// Who reads a family's outputs. The host reads a selection as its step
/// completes, with the next step already queued: read from device memory, it
/// would queue a copy behind that step and hold the host until it ends, so
/// no step would ever be queued ahead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Outputs {
    /// Device work reads them: `DeviceOnly`.
    Device,
    /// The host reads them: `Shared`.
    Host,
}

impl NativeGraphPool {
    /// The startup slots of `charge`: its activations in `arena`, each with
    /// its upload regions, and its startup output slots. Each activation's
    /// slot is a launch in flight of its family's graph hold
    /// (`seismic::NativeGraphFamily::graphs`).
    pub(crate) fn new(
        domain: ResourceDomainId,
        family: &NativeGraphFamily,
        charge: &crate::NativeGraphCharge,
        arena: &NativeExecutionArena,
        outputs: Outputs,
    ) -> Result<Self, super::AllocationError> {
        let floor = GraphSlots {
            activations: charge.activations,
            output: charge.output_slots,
        };
        let mut pool = Self {
            domain,
            family: family.clone(),
            arena: arena.clone(),
            upload_regions: charge.upload_regions.clone(),
            activation_bytes: charge.activation_bytes(),
            output_bytes: charge.output_bytes,
            outputs,
            floor,
            held: floor,
            allocated: GraphSlots {
                activations: 0,
                output: 0,
            },
            workspace: Rc::new(RefCell::new(Vec::new())),
            output: Rc::new(RefCell::new(Vec::new())),
        };
        pool.grow(floor)?;
        Ok(pool)
    }

    /// The bytes [`Self::grow`] of `slots` allocates.
    pub(crate) fn hold_bytes(&self, slots: GraphSlots) -> Result<u64, &'static str> {
        class_bytes(
            self.activation_bytes,
            slots.activations.saturating_sub(self.allocated.activations),
            self.output_bytes,
            slots.output.saturating_sub(self.allocated.output),
        )
    }

    /// What the activations up to `slots`' count beyond those allocated
    /// add to the device's graph budget: the family's hold at the larger
    /// count beyond its hold at those allocated, each a launch in flight
    /// (`seismic::LaunchGraphs::held`), which grows as they are allocated
    /// ([`Self::grow`]).
    pub(crate) fn hold_graphs(&self, slots: GraphSlots) -> Option<seismic::GraphCensus> {
        let graphs = self.family.graphs();
        let held = u64::try_from(self.allocated.activations).ok()?;
        let grown = u64::try_from(slots.activations.max(self.allocated.activations)).ok()?;
        graphs.held(grown)?.minus(graphs.held(held)?)
    }

    /// Allocate activations and output slots up to `slots`' counts, held
    /// only by [`Self::hold`]. The caller holds a heap claim for
    /// [`Self::hold_bytes`] across this call. On a failure what was
    /// allocated idles until released.
    pub(crate) fn grow(&mut self, slots: GraphSlots) -> Result<(), super::AllocationError> {
        let grows = self.allocated.activations < slots.activations;
        while self.allocated.activations < slots.activations {
            let slot = self
                .family
                .new_slot_in(&self.arena, &self.upload_regions)
                .map_err(super::AllocationError::Workflow)?;
            self.workspace.borrow_mut().push(Claimed::under_class(slot));
            self.allocated.activations += 1;
        }
        // Each launch in flight's decode graphs, recorded as the slots are
        // made, at load and as a class grows, while memory serves them: no
        // later step records them, a bank shrink's under pressure included.
        if grows {
            self.family.record_decode().map_err(super::AllocationError::Tensor)?;
        }
        while self.allocated.output < slots.output {
            let slot = self.new_output().map_err(super::AllocationError::Tensor)?;
            self.push_output(Claimed::under_class(slot));
        }
        Ok(())
    }

    /// Hold `slots`' class, which [`Self::grow`] allocated: its slots are
    /// kept until Reclaim releases the class.
    pub(crate) fn hold(&mut self, slots: GraphSlots) {
        self.held = GraphSlots {
            activations: self.held.activations.max(slots.activations),
            output: self.held.output.max(slots.output),
        };
    }

    /// Hold the floor's slots alone again: those beyond idle until
    /// released.
    pub(crate) fn hold_floor(&mut self) {
        self.held = self.floor;
    }

    pub fn domain(&self) -> &ResourceDomainId {
        &self.domain
    }

    /// The slots of a group of one place, never released.
    pub(crate) fn floor(&self) -> GraphSlots {
        self.floor
    }

    /// Physical arenas owned by this pool, including slots currently lent
    /// to a submission or a published output view.
    pub fn committed_bytes(&self) -> Result<u64, &'static str> {
        class_bytes(
            self.activation_bytes,
            self.allocated.activations,
            self.output_bytes,
            self.allocated.output,
        )
    }

    /// The charge of one more output slot.
    pub fn output_slot_bytes(&self) -> u64 {
        self.output_bytes
    }

    pub fn available_workspace(&self) -> usize {
        self.workspace.borrow().len()
    }

    pub fn available_output(&self) -> usize {
        self.output.borrow().len()
    }

    /// Allocate one more output slot under `held`, the claim of its
    /// [`Self::output_slot_bytes`], kept until the slot is released.
    pub(crate) fn grow_claimed_output(&mut self, held: crate::memory::ledger::Held) -> Result<(), seismic::TensorError> {
        let slot = self.new_output()?;
        self.push_output(Claimed::under(slot, held));
        Ok(())
    }

    fn new_output(&self) -> Result<NativeGraphFamilyOutputSlot, seismic::TensorError> {
        match self.outputs {
            Outputs::Device => self.family.new_output_slot(),
            Outputs::Host => self.family.new_host_output_slot(),
        }
    }

    fn push_output(&mut self, slot: Claimed<NativeGraphFamilyOutputSlot>) {
        self.output.borrow_mut().push(slot);
        self.allocated.output += 1;
    }

    /// Take the free activations and output slots beyond the held class's
    /// counts into `idle`, which releases them where it drops. Lent slots
    /// stay until their last view drops.
    pub(crate) fn release_idle(&mut self, idle: &mut IdleSlots) {
        fn release<T>(free: &mut Vec<T>, allocated: &mut usize, kept: usize, into: &mut Vec<T>) {
            let releasable = allocated.saturating_sub(kept).min(free.len());
            into.extend(free.drain(free.len() - releasable..));
            *allocated -= releasable;
        }
        release(&mut self.workspace.borrow_mut(), &mut self.allocated.activations, self.held.activations, &mut idle.activations);
        release(&mut self.output.borrow_mut(), &mut self.allocated.output, self.held.output, &mut idle.outputs);
    }

    pub fn acquire_workspace(&self) -> Result<NativeGraphWorkspaceLease, CapacityError> {
        let slot = self.workspace.borrow_mut().pop().ok_or(CapacityError {
            resource: ResourceKind::Workspace,
            required: 1,
            available: 0,
        })?;
        Ok(NativeGraphWorkspaceLease {
            domain: self.domain.clone(),
            slot: Some(slot),
            free: self.workspace.clone(),
        })
    }

    pub fn acquire_output(&self) -> Result<NativeGraphOutputLease, CapacityError> {
        let slot = self.output.borrow_mut().pop().ok_or(CapacityError {
            resource: ResourceKind::Output,
            required: 1,
            available: 0,
        })?;
        let Claimed { slot, claims } = slot;
        Ok(NativeGraphOutputLease {
            domain: self.domain.clone(),
            slot: Some(slot),
            claims,
            free: self.output.clone(),
        })
    }
}

/// Vision storage, one family per class: an encode's workspace and its
/// image's output slot are of its image's class, so a small image never holds
/// the largest class's storage. Vision holds nothing until an image arrives;
/// an encode claims an activation and an output slot of its class under the
/// heap, the image holds its slot until its features are released, and free
/// slots are released when idle.
pub struct VisionGraphPool {
    domain: ResourceDomainId,
    classes: Vec<VisionGraphClass>,
}

struct VisionGraphClass {
    patch_rows: u64,
    family: NativeGraphFamily,
    activations: usize,
    outputs: usize,
    workspace: FreeSlots<NativeGraphFamilySlot>,
    output: FreeSlots<NativeGraphFamilyOutputSlot>,
}

impl VisionGraphClass {
    /// An encode runs one graph at a time to completion.
    fn upload_regions(&self) -> Vec<u64> {
        crate::planning::serial_upload_regions(self.family.upload_bytes())
    }

    /// One activation's own workspace and its upload regions.
    fn activation_bytes(&self) -> u64 {
        self.family.workspace_bytes() + self.upload_regions().iter().sum::<u64>()
    }
}

impl VisionGraphPool {
    pub(crate) fn new<'a>(
        domain: ResourceDomainId,
        classes: impl IntoIterator<Item = (u64, &'a NativeGraphFamily)>,
    ) -> Self {
        Self {
            domain,
            classes: classes
                .into_iter()
                .map(|(patch_rows, family)| VisionGraphClass {
                    patch_rows,
                    family: family.clone(),
                    activations: 0,
                    outputs: 0,
                    workspace: Rc::new(RefCell::new(Vec::new())),
                    output: Rc::new(RefCell::new(Vec::new())),
                })
                .collect(),
        }
    }

    fn class(&self, patch_rows: u64) -> Result<&VisionGraphClass, CapacityError> {
        self.classes
            .iter()
            .find(|class| class.patch_rows == patch_rows)
            .ok_or(CapacityError {
                resource: ResourceKind::Workspace,
                required: 1,
                available: 0,
            })
    }

    fn class_mut(&mut self, patch_rows: u64) -> Result<&mut VisionGraphClass, CapacityError> {
        self.classes
            .iter_mut()
            .find(|class| class.patch_rows == patch_rows)
            .ok_or(CapacityError {
                resource: ResourceKind::Workspace,
                required: 1,
                available: 0,
            })
    }

    /// The charge of one activation of the class of `patch_rows`: its own
    /// workspace and its upload regions.
    pub fn activation_bytes(&self, patch_rows: u64) -> Result<u64, CapacityError> {
        Ok(self.class(patch_rows)?.activation_bytes())
    }

    /// What the next activation of the class of `patch_rows` adds to the
    /// device's graph budget: the class family's hold at one activation more
    /// beyond its hold now (`seismic::LaunchGraphs::held`).
    pub fn activation_graphs(&self, patch_rows: u64) -> Result<seismic::GraphCensus, CapacityError> {
        let class = self.class(patch_rows)?;
        let graphs = class.family.graphs();
        let held = class.activations as u64;
        Ok(graphs
            .held(held + 1)
            .and_then(|grown| grown.minus(graphs.held(held)?))
            .expect("a vision class's graph census fits u64"))
    }

    /// The charge of one output slot of the class of `patch_rows`.
    pub fn output_bytes(&self, patch_rows: u64) -> Result<u64, CapacityError> {
        Ok(self.class(patch_rows)?.family.output_bytes())
    }

    pub fn available_workspace(&self, patch_rows: u64) -> usize {
        self.class(patch_rows)
            .map_or(0, |class| class.workspace.borrow().len())
    }

    pub fn available_output(&self, patch_rows: u64) -> usize {
        self.class(patch_rows)
            .map_or(0, |class| class.output.borrow().len())
    }

    /// Allocate one more activation of the class of `patch_rows` under
    /// `held`, the claim of its [`Self::activation_bytes`] and graphs, kept
    /// until the activation is released.
    pub(crate) fn grow_activation(&mut self, patch_rows: u64, held: crate::memory::ledger::Held) -> Result<(), VisionGrowth> {
        let class = self.class_mut(patch_rows).map_err(VisionGrowth::Class)?;
        let slot = class
            .family
            .new_slot(&class.upload_regions())
            .map_err(VisionGrowth::Device)?;
        class.workspace.borrow_mut().push(Claimed::under(slot, held));
        class.activations += 1;
        Ok(())
    }

    /// Allocate one more output slot of the class of `patch_rows` under
    /// `held`, the claim of its [`Self::output_bytes`], kept until the slot
    /// is released.
    pub(crate) fn grow_output(&mut self, patch_rows: u64, held: crate::memory::ledger::Held) -> Result<(), VisionGrowth> {
        let class = self.class_mut(patch_rows).map_err(VisionGrowth::Class)?;
        let slot = class
            .family
            .new_output_slot()
            .map_err(VisionGrowth::Device)?;
        class.output.borrow_mut().push(Claimed::under(slot, held));
        class.outputs += 1;
        Ok(())
    }

    pub fn acquire_workspace(
        &self,
        patch_rows: u64,
    ) -> Result<NativeGraphWorkspaceLease, CapacityError> {
        let class = self.class(patch_rows)?;
        let slot = class.workspace.borrow_mut().pop().ok_or(CapacityError {
            resource: ResourceKind::Workspace,
            required: 1,
            available: 0,
        })?;
        Ok(NativeGraphWorkspaceLease {
            domain: self.domain.clone(),
            slot: Some(slot),
            free: class.workspace.clone(),
        })
    }

    pub fn acquire_output(&self, patch_rows: u64) -> Result<NativeGraphOutputLease, CapacityError> {
        let class = self.class(patch_rows)?;
        let slot = class.output.borrow_mut().pop().ok_or(CapacityError {
            resource: ResourceKind::Output,
            required: 1,
            available: 0,
        })?;
        let Claimed { slot, claims } = slot;
        Ok(NativeGraphOutputLease {
            domain: self.domain.clone(),
            slot: Some(slot),
            claims,
            free: class.output.clone(),
        })
    }

    /// Take every free activation and output slot into `idle`, which
    /// releases them where it drops. A lent activation stays until its
    /// encode completes, a lent output slot until its image's features are
    /// released.
    pub(crate) fn release_idle(&mut self, idle: &mut IdleSlots) {
        for class in &mut self.classes {
            let (activations, outputs) = (idle.activations.len(), idle.outputs.len());
            idle.activations.extend(class.workspace.borrow_mut().drain(..));
            idle.outputs.extend(class.output.borrow_mut().drain(..));
            class.activations -= idle.activations.len() - activations;
            class.outputs -= idle.outputs.len() - outputs;
        }
    }

    /// Storage allocated now, lent or free.
    pub fn committed_bytes(&self) -> Result<u64, &'static str> {
        self.classes.iter().try_fold(0u64, |bytes, class| {
            class_bytes(
                class.activation_bytes(),
                class.activations,
                class.family.output_bytes(),
                class.outputs,
            )?
            .checked_add(bytes)
            .ok_or(OVERFLOW)
        })
    }
}

const OVERFLOW: &str = "a graph pool's bytes overflow";

/// Free graph slots the pools gave up, which release their storage (each
/// slot's upload regions and record tables, and its family's graphs beyond
/// the slots left) as they are released ([`Self::release`]): on the load's
/// release thread (`AllocatedResources::release_idle_slots`).
#[derive(Default)]
pub(crate) struct IdleSlots {
    activations: Vec<Claimed<NativeGraphFamilySlot>>,
    outputs: Vec<Claimed<NativeGraphFamilyOutputSlot>>,
}

impl IdleSlots {
    pub(crate) fn len(&self) -> usize {
        self.activations.len() + self.outputs.len()
    }

    /// Release every slot: the bytes the device is no longer charged.
    pub(crate) fn release(self) -> u64 {
        let activations = self.activations.into_iter().map(|slot| slot.release(NativeGraphFamilySlot::release));
        activations.chain(self.outputs.into_iter().map(|slot| slot.release(NativeGraphFamilyOutputSlot::release))).sum()
    }
}

/// A pool's free slots.
type FreeSlots<S> = Rc<RefCell<Vec<Claimed<S>>>>;

/// A slot and the claims of its storage, which go where it goes and are
/// released with it: a slot grown one at a time holds its own; one of a
/// held class holds none, its class's claim paying for it.
pub(crate) struct Claimed<S> {
    slot: S,
    claims: Vec<crate::memory::ledger::Claim>,
}

impl<S> Claimed<S> {
    /// A slot just allocated under `held`.
    fn under(slot: S, mut held: crate::memory::ledger::Held) -> Self {
        held.allocated();
        Self { slot, claims: held.into_claims() }
    }

    /// A slot of a held class, under its class's claim.
    fn under_class(slot: S) -> Self {
        Self { slot, claims: Vec::new() }
    }

    /// Release the slot's storage by `release`, then its claims.
    fn release(self, release: impl FnOnce(S) -> u64) -> u64 {
        let Self { slot, claims } = self;
        let released = release(slot);
        drop(claims);
        released
    }
}


/// A failed vision slot growth: a class the load did not prepare, or the
/// device's refusal.
pub(crate) enum VisionGrowth {
    Class(CapacityError),
    Device(seismic::TensorError),
}

pub struct NativeGraphWorkspaceLease {
    domain: ResourceDomainId,
    slot: Option<Claimed<NativeGraphFamilySlot>>,
    free: FreeSlots<NativeGraphFamilySlot>,
}

impl NativeGraphWorkspaceLease {
    pub fn domain(&self) -> &ResourceDomainId {
        &self.domain
    }

    pub fn slot_mut(&mut self) -> &mut NativeGraphFamilySlot {
        &mut self.slot.as_mut().expect("graph workspace lease is live").slot
    }
}

impl Drop for NativeGraphWorkspaceLease {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.take() {
            self.free.borrow_mut().push(slot);
        }
    }
}

/// An output reservation owns one family arena. Activation transfers it into
/// a one-shot Seismic output value; successful work returns that value after
/// every exported tensor view has been released.
pub struct NativeGraphOutputLease {
    domain: ResourceDomainId,
    slot: Option<NativeGraphFamilyOutputSlot>,
    /// The slot's claims, kept while it is activated.
    claims: Vec<crate::memory::ledger::Claim>,
    free: FreeSlots<NativeGraphFamilyOutputSlot>,
}

impl NativeGraphOutputLease {
    pub fn domain(&self) -> &ResourceDomainId {
        &self.domain
    }

    pub fn activate(
        &mut self,
        plan: &NativeGraphPlan,
    ) -> Result<NativeGraphOutputs, InvariantError> {
        let slot = self.slot.take().expect("graph output lease is live");
        slot.activate(plan).map_err(|error| InvariantError {
            context: "target graph output",
            detail: error.to_string(),
        })
    }

    pub fn recycle(&mut self, outputs: NativeGraphOutputs) -> Result<(), InvariantError> {
        if self.slot.is_some() {
            return Err(InvariantError {
                context: "target graph output",
                detail: "output arena was already recycled".into(),
            });
        }
        self.slot = Some(outputs.recycle().map_err(|error| InvariantError {
            context: "target graph output",
            detail: error.to_string(),
        })?);
        Ok(())
    }

    /// Publish a completed graph's outputs. Every exported tensor returned
    /// through this owner retains the physical output arena. Final drop
    /// recycles it into the same admitted pool.
    pub fn publish(mut self, outputs: NativeGraphOutputs) -> GraphOutputOwner {
        assert!(self.slot.is_none(), "graph output was not activated");
        GraphOutputOwner {
            claim: Rc::new(GraphOutputClaim {
                outputs: Some(outputs),
                claims: std::mem::take(&mut self.claims),
                free: self.free.clone(),
            }),
        }
    }
}

impl Drop for NativeGraphOutputLease {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.take() {
            self.free.borrow_mut().push(Claimed { slot, claims: std::mem::take(&mut self.claims) });
        }
    }
}

struct GraphOutputClaim {
    outputs: Option<NativeGraphOutputs>,
    claims: Vec<crate::memory::ledger::Claim>,
    free: FreeSlots<NativeGraphFamilyOutputSlot>,
}

impl Drop for GraphOutputClaim {
    fn drop(&mut self) {
        let Some(outputs) = self.outputs.take() else {
            return;
        };
        let slot = outputs
            .recycle()
            .expect("a graph output view outlived its last ownership claim");
        self.free.borrow_mut().push(Claimed { slot, claims: std::mem::take(&mut self.claims) });
    }
}

#[derive(Clone)]
pub struct GraphOutputOwner {
    claim: Rc<GraphOutputClaim>,
}

impl GraphOutputOwner {
    pub fn tensor(&self, result: &WorkflowTensor) -> Option<GraphOutputTensor> {
        let tensor = self.claim.outputs.as_ref()?.exported(result)?;
        Some(GraphOutputTensor {
            tensor,
            claim: self.claim.clone(),
        })
    }
}

pub struct GraphOutputTensor {
    // Rust drops fields in declaration order. Release this tensor handle
    // before the final claim attempts to recycle the output arena.
    tensor: Tensor,
    claim: Rc<GraphOutputClaim>,
}

impl Clone for GraphOutputTensor {
    fn clone(&self) -> Self {
        Self {
            tensor: self.tensor.clone(),
            claim: self.claim.clone(),
        }
    }
}

impl GraphOutputTensor {
    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    pub fn slice_leading(&self, start: u64, end: u64) -> Result<Self, seismic::TensorError> {
        Ok(Self {
            tensor: self.tensor.slice_leading(start, end)?,
            claim: self.claim.clone(),
        })
    }
}

pub type TargetGraphPool = NativeGraphPool;
pub type TargetGraphWorkspaceLease = NativeGraphWorkspaceLease;
pub type TargetGraphOutputLease = NativeGraphOutputLease;

#[cfg(test)]
mod tests {
    /// A class's bytes are its slots' bytes, and refused where either
    /// product or their sum leaves `u64`.
    #[test]
    fn a_class_refuses_bytes_beyond_u64() {
        assert_eq!(super::class_bytes(3, 2, 5, 4), Ok(26));
        assert!(super::class_bytes(u64::MAX, 2, 0, 0).is_err());
        assert!(super::class_bytes(0, 0, u64::MAX / 2 + 1, 2).is_err());
        assert!(super::class_bytes(u64::MAX, 1, 1, 1).is_err());
    }

    /// A slot a pool frees keeps its claim until the slot's storage is
    /// released, and gives it back then: a freed vision slot leaves nothing
    /// held on the ledger.
    #[test]
    fn a_freed_slots_claim_is_released_with_it() {
        use crate::memory::{ClaimDemand, ledger::{CommittedKind, DeviceLedgers, Ledgers}};
        let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
        let device = crate::test_support::open_backend(&catalog, seismic::BackendName::Cpu).expect("the CPU backend opens");
        let shape = std::sync::Arc::new(crate::memory::ledger::LoadShape { stores: Vec::new() });
        let ledgers = DeviceLedgers::open_in(&Ledgers::new(), &device, shape);
        let at = std::time::Instant::now();
        for ledger in [ledgers.device(), ledgers.host()] {
            ledger.read(|| Ok(crate::memory::ledger::Available::new(8 << 30, 8 << 30, None, at).expect("a reading")));
        }
        let held = ledgers.claim(&device, CommittedKind::GroupClass, ClaimDemand { device: 1 << 20, ..Default::default() }).expect("room");
        let slot = super::Claimed::under((), held);
        assert_eq!(ledgers.device().snapshot().allocated(), 1 << 20, "the freed slot's claim is held until its release");
        assert_eq!(slot.release(|()| 0), 0);
        assert_eq!(ledgers.device().snapshot().committed(), 0, "the released slot's claim is gone");
        assert_eq!(ledgers.device().snapshot().allocated(), 0);
    }
}
