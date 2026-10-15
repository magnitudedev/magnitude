//! A streamed target's banks as a load holds them: the device tensors the
//! graphs bind and the watcher that serves them (spec 4.3–4.7).

use super::layout::BankTensors;
use super::runtime::{
    Heartbeat, ScratchFill, StagingPool, StoredTensor, StreamedLayer, StreamingRuntime,
};
use super::source::{ExpertSource, ModelFile};
use super::words::{SharedBytes, SharedWords};
use crate::ResidentRoles;
use crate::planning::{BANK_KINDS, ExpertBankPlan};
use crate::residency::{ImportArtifactTensor, Stored};
use magnitude_artifacts::gguf::GgufArtifact;
use magnitude_family_contracts::WeightRole;
use seismic::Device;
use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};

/// The resident side of a streamed target: every bank starts empty (rule 5).
pub(crate) struct ResidentBank {
    pub tensors: BankTensors,
    runtime: StreamingRuntime,
    /// The sequence of the last bank-served step submitted: the device's
    /// step word as `routed_step` will have advanced it (4.3).
    submitted: AtomicI32,
}

/// One submitted bank-served step. Completing it reports the watcher's
/// failure of any of its requests (rule 6); a step dropped unfinished is
/// still counted complete, so the watcher stops polling for it.
pub(crate) struct BankStep {
    bank: Arc<ResidentBank>,
    sequence: i32,
    completed: bool,
}

impl BankStep {
    pub(crate) fn complete(mut self) -> Result<(), String> {
        self.completed = true;
        self.bank.runtime.step_completed(self.sequence, true)
    }
}

impl Drop for BankStep {
    fn drop(&mut self) {
        if !self.completed {
            let _ = self.bank.runtime.step_completed(self.sequence, false);
        }
    }
}

impl std::fmt::Debug for BankStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BankStep")
            .field("sequence", &self.sequence)
            .finish()
    }
}

impl ResidentBank {
    /// Allocate the banks' tensors and start the watcher over `target`'s
    /// stored expert records. `sublayers` holds the empty bank weights,
    /// whose descriptors name the stored tensors.
    pub(crate) fn open(
        device: &Device,
        plan: &ExpertBankPlan,
        target: &GgufArtifact,
        sublayers: &ResidentRoles,
    ) -> Result<Self, String> {
        let shared = plan.shared.as_ref().ok_or("a streamed load without its shared bank plan")?;
        let tensors = BankTensors::allocate(device, plan, shared)?;
        let directory = super::shared_directory::SharedDirectory::new(shared, plan.experts)
            .map_err(|e| format!("shared bank directory: {e:?}"))?;
        let mut path = None;
        let mut layers = Vec::with_capacity(plan.layers.len());
        for (layer, bank) in plan.layers.iter().zip(&tensors.layers) {
            let mut stored = [StoredTensor {
                base: 0,
                bytes: 0,
                pool: 0,
            }; 3];
            for (kind, planned) in layer.tensors.iter().enumerate() {
                let role = WeightRole {
                    scope: layer.scope,
                    kind: BANK_KINDS[kind],
                };
                let descriptor = sublayers.get(role)?.descriptor();
                let source = ImportArtifactTensor::from_gguf(target, descriptor)
                    .map_err(|error| error.to_string())?;
                let Stored::GgmlBlocks { .. } = source.stored() else {
                    return Err(format!("{role:?} is not stored as packed records"));
                };
                let (file, offset, bytes) = source.stored().file_range();
                if Some(bytes) != planned.stored_bytes.checked_mul(plan.experts) {
                    return Err(format!(
                        "{role:?} stores {bytes} bytes, not {} experts of {}",
                        plan.experts, planned.stored_bytes
                    ));
                }
                match &path {
                    None => path = Some(file.path().to_owned()),
                    Some(path) if path == file.path() => {}
                    Some(_) => return Err("streamed experts span several files".into()),
                }
                stored[kind] = StoredTensor {
                    base: offset,
                    bytes: planned.stored_bytes,
                    pool: planned.pool,
                };
            }
            let words = |tensor: &seismic::Tensor| SharedWords::new(tensor.clone());
            layers.push(StreamedLayer::new(
                words(&bank.slot_map)?,
                words(&bank.stamps)?,
                words(&bank.commits)?,
                words(&bank.control)?,
                words(&bank.placement)?,
                words(&bank.scratch_control)?,
                [words(&bank.rings[0])?, words(&bank.rings[1])?],
                stored,
            ));
        }
        SharedWords::new(tensors.step.clone())?.fill(0);
        let records =
            u32::try_from(plan.staging_records).map_err(|_| "staging records exceed u32")?;
        let prefetch = u32::try_from(crate::streaming::layout::prefetch_records(plan.selected))
            .map_err(|_| "prefetch records exceed u32")?;
        let pools = plan
            .pools
            .iter()
            .zip(&tensors.pools)
            .map(|(pool, tensor)| {
                Ok(StagingPool::new(
                    SharedBytes::new(tensor.clone())?,
                    records,
                    prefetch,
                    pool.record_bytes,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let path = path.ok_or("a streamed load without routed layers")?;
        let source = ExpertSource::new(ModelFile::open(&path)?)?;
        let words = |tensor: &seismic::Tensor| SharedWords::new(tensor.clone());
        let scratch = ScratchFill::new(
            words(&tensors.scratch_flags)?,
            tensors
                .scratch_lists
                .iter()
                .map(|[gate, up, down]| Ok([words(gate)?, words(up)?, words(down)?]))
                .collect::<Result<Vec<_>, String>>()?,
            {
                let [gate, up, down] = &tensors.copy_lists;
                [words(gate)?, words(up)?, words(down)?]
            },
            usize::try_from(plan.experts).map_err(|_| "experts exceed the host")?,
            usize::try_from(plan.round_records()).map_err(|_| "round records exceed the host")?,
        );
        let heartbeat = Heartbeat::new(words(&tensors.heartbeat)?);
        let runtime = StreamingRuntime::new(layers, scratch, heartbeat, pools, source, directory)?;
        Ok(Self {
            tensors,
            runtime,
            submitted: AtomicI32::new(0),
        })
    }

    /// A step is being submitted, bank-served or scratch-served (spec 4.5);
    /// steps run in submission order, so its sequence is the next one.
    pub(crate) fn begin_step(self: &Arc<Self>) -> BankStep {
        let sequence = self.submitted.fetch_add(1, Ordering::AcqRel) + 1;
        self.runtime.step_submitted();
        BankStep {
            bank: Arc::clone(self),
            sequence,
            completed: false,
        }
    }
}
