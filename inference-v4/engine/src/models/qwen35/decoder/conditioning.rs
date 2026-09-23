//! Conditioned-input overlays use the same generated, device-specialized
//! kernel for every runtime shape. Shape specialization is inferred from the
//! tensors at the call boundary; the engine owns no compiler envelope.

use crate::{kernels, models::qwen35::inputs::Assembled, Error};
use seismic::{Device, Kernel, PreparationOptions, Tensor, WorkflowDraft, WorkflowTensor};

pub(super) struct Overlay {
    pub(super) source: Tensor,
    pub(super) destination: usize,
    pub(super) count: usize,
}

pub(super) struct Conditioning {
    kernel: Kernel<kernels::qwen_conditioning_overlay::Entry>,
    hidden: u64,
}

impl Conditioning {
    pub(super) fn new(
        device: &Device,
        preparation: PreparationOptions,
        hidden: u64,
    ) -> Result<Self, Error> {
        Ok(Self {
            kernel: kernels::qwen_conditioning_overlay::for_device(device, preparation)?,
            hidden,
        })
    }

    pub(super) fn prepare(&self, assembled: &Assembled) -> Result<Vec<Overlay>, Error> {
        assembled
            .features
            .iter()
            .map(|feature| {
                if feature.source.extents() != [feature.count as u64, self.hidden] {
                    return Err("conditioned feature has the wrong tensor shape".into());
                }
                Ok(Overlay {
                    source: feature.source.clone(),
                    destination: feature.destination,
                    count: feature.count,
                })
            })
            .collect()
    }

    pub(super) fn enqueue(
        &self,
        workflow: &mut WorkflowDraft,
        overlay: &Overlay,
        hidden: &WorkflowTensor,
    ) -> Result<(), Error> {
        let start = u64::try_from(overlay.destination)
            .map_err(|_| "feature destination exceeds the Seismic shape domain")?;
        let end = start
            .checked_add(
                u64::try_from(overlay.count)
                    .map_err(|_| "feature length exceeds the Seismic shape domain")?,
            )
            .ok_or("feature destination overflow")?;
        let mut destination = hidden.slice_leading(start, end);
        let _ = workflow.enqueue(
            &self.kernel,
            kernels::qwen_conditioning_overlay::WorkflowArgs {
                input: (&overlay.source).into(),
                out: (&mut destination).into(),
            },
        )?;
        Ok(())
    }
}
