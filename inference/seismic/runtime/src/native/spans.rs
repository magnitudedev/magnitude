//! The device time of spans of a graph run: a plan's marks
//! (`NativeGraphDraft::span_mark`, each a boundary of its calls), stamped on
//! the device as a sampled run passes them
//! (`ReadyNativeGraphRun::queue_sampled`), and read once its work has
//! completed ([`NativeSpans::seconds`]). How lines observe the real work they
//! price. A mark divides no graph and adds none, and a mark at a graph's edge
//! (a run's ends, a timeline node) is stamped between graphs only on a
//! sampled run.
//!
//! Inside a graph, CUDA holds an event-record node in the graph's chain,
//! which a sampled run sets to its own events; the driver can't disable one,
//! so a run not sampled pays that one stamp into the graph's own event.
//! Vulkan holds a boundary between the graph's secondary command buffers,
//! which a run not sampled executes as one `vkCmdExecuteCommands` and pays
//! nothing for, and a sampled run writes `vkCmdWriteTimestamp2` into the
//! run's own pool there, in its primary command buffer. Metal and the CPU
//! have no span timing.

use crate::api::device::DeviceInner;
use crate::api::{CallError, WorkflowError};
use crate::backends::OpenedKind;
use seismic_compiler::errors::ExecutionError;
use std::sync::Arc;

/// A sampled run's stamps at its plan's marks, by mark ordinal.
#[derive(Clone)]
pub struct NativeSpans {
    stamps: Arc<Stamps>,
    marks: usize,
}

/// The backend's stamps.
pub(crate) enum Stamps {
    Cuda(seismic_cuda::direct::Stamps),
    #[cfg(not(target_os = "macos"))]
    Vulkan(seismic_vulkan::direct::Stamps),
}

/// One mark of a sampled run, stamped at its boundary.
#[derive(Clone)]
pub(crate) struct Stamp {
    pub(crate) stamps: Arc<Stamps>,
    pub(crate) mark: usize,
}

impl NativeSpans {
    /// The stamps of `marks` marks on `device`, or why its backend times no
    /// span.
    pub(crate) fn new(device: &DeviceInner, marks: usize) -> Result<Self, CallError> {
        let stamps = match &device.kind {
            OpenedKind::Cuda(opened) => seismic_cuda::direct::Stamps::new(opened.service(), marks).map(Stamps::Cuda),
            #[cfg(not(target_os = "macos"))]
            OpenedKind::Vulkan(opened) => seismic_vulkan::direct::Stamps::new(opened.service(), marks).map(Stamps::Vulkan),
            kind => {
                return Err(CallError::Workflow(WorkflowError::SpanTimingUnsupported {
                    backend: super::backend_name(kind),
                }))
            }
        }
        .map_err(CallError::Execution)?;
        Ok(Self { stamps: Arc::new(stamps), marks })
    }

    /// Each mark's stamp, in mark order.
    pub(crate) fn stamps(&self) -> impl Iterator<Item = Stamp> + '_ {
        (0..self.marks).map(|mark| Stamp { stamps: self.stamps.clone(), mark })
    }

    /// The marks it stamps.
    pub fn marks(&self) -> usize {
        self.marks
    }

    /// The device seconds from mark `from` to mark `to` of the run, once its
    /// work has passed `to` (waited for here: read it once the run's
    /// completion has).
    pub fn seconds(&self, from: usize, to: usize) -> Result<f64, ExecutionError> {
        assert!(from <= to && to < self.marks, "marks {from} to {to} of a run of {} marks", self.marks);
        match &*self.stamps {
            Stamps::Cuda(stamps) => stamps.seconds(from, to),
            #[cfg(not(target_os = "macos"))]
            Stamps::Vulkan(stamps) => stamps.seconds(from, to),
        }
    }
}
