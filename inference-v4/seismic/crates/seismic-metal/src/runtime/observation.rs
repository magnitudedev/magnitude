//! Optional native qualification timestamps. No selection or model construction
//! consumes this module. One sampled compute encoder identifies each dispatch.
use super::*;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLCommonCounterSetTimestamp, MTLComputePassDescriptor, MTLCounterSampleBuffer,
    MTLCounterSampleBufferDescriptor, MTLCounterSamplingPoint, MTLCounterSet, MTLStorageMode,
};

#[derive(Clone, Debug)]
pub struct DispatchObservation {
    pub invocation: usize,
    pub launch: usize,
    pub kernel: String,
    /// Capture-relative calibrated GPU clock, not a host wall-clock timestamp.
    pub started_ns: u64,
    pub elapsed_ns: u64,
}
#[derive(Clone, Debug)]
pub struct Observation {
    pub command_seconds: f64,
    pub dispatches: Vec<DispatchObservation>,
}
impl Observation {
    pub const CLOCK: &'static str = "metal-stage-timestamps-capture-relative";
}

pub(super) struct Capture {
    samples: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    start: (u64, u64),
    dispatches: Vec<(usize, usize, String)>,
    capacity: usize,
}
fn timestamps(device: &ProtocolObject<dyn MTLDevice>) -> (u64, u64) {
    let (mut cpu, mut gpu) = (0, 0);
    // Both stack locations live through this synchronous device query.
    unsafe {
        device.sampleTimestamps_gpuTimestamp(NonNull::from(&mut cpu), NonNull::from(&mut gpu));
    }
    (cpu, gpu)
}
impl Capture {
    pub(super) fn new(device: &Device, count: usize) -> Result<Self, String> {
        if !device
            .device
            .supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary)
        {
            return Err("Metal stage timestamp sampling is unavailable on this device".into());
        }
        let count = count
            .checked_mul(2)
            .filter(|&n| n > 0)
            .ok_or("invalid Metal timestamp sample count")?;
        let sets = device
            .device
            .counterSets()
            .ok_or("Metal counter sets are unavailable")?;
        let set = sets
            .iter()
            .find(|set| &*set.name() == unsafe { MTLCommonCounterSetTimestamp })
            .ok_or("Metal timestamp counter set is unavailable")?;
        let descriptor = MTLCounterSampleBufferDescriptor::new();
        descriptor.setCounterSet(Some(&set));
        descriptor.setStorageMode(MTLStorageMode::Shared);
        descriptor.setLabel(&NSString::from_str("Seismic dispatch qualification"));
        // Every attachment index is checked against this allocated sample count.
        unsafe {
            descriptor.setSampleCount(count);
        }
        let samples = device
            .device
            .newCounterSampleBufferWithDescriptor_error(&descriptor)
            .map_err(|e| {
                format!(
                    "Metal timestamp allocation failed: {}",
                    e.localizedDescription()
                )
            })?;
        Ok(Self {
            samples,
            start: timestamps(&device.device),
            dispatches: Vec::new(),
            capacity: count / 2,
        })
    }
    pub(super) fn encoder(
        &mut self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        invocation: usize,
        launch: usize,
        kernel: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>, String> {
        let index = self.dispatches.len();
        if index >= self.capacity {
            return Err("Metal timestamp capture exceeded its dispatch count".into());
        }
        let descriptor = MTLComputePassDescriptor::new();
        let attachment = unsafe {
            descriptor
                .sampleBufferAttachments()
                .objectAtIndexedSubscript(0)
        };
        attachment.setSampleBuffer(Some(&self.samples));
        unsafe {
            attachment.setStartOfEncoderSampleIndex(index * 2);
            attachment.setEndOfEncoderSampleIndex(index * 2 + 1);
        }
        let encoder = command
            .computeCommandEncoderWithDescriptor(&descriptor)
            .ok_or("could not create timestamped Metal encoder")?;
        self.dispatches.push((invocation, launch, kernel.into()));
        Ok(encoder)
    }
    pub(super) fn finish(self, device: &Device) -> Result<Vec<DispatchObservation>, String> {
        let end = timestamps(&device.device);
        if self.dispatches.len() != self.capacity {
            return Err("incomplete Metal dispatch capture".into());
        }
        let cpu_delta = end
            .0
            .checked_sub(self.start.0)
            .filter(|&n| n > 0)
            .ok_or("invalid Metal CPU calibration interval")?;
        let gpu_delta = end
            .1
            .checked_sub(self.start.1)
            .filter(|&n| n > 0)
            .ok_or("invalid Metal GPU calibration interval")?;
        let count = self
            .dispatches
            .len()
            .checked_mul(2)
            .ok_or("Metal timestamp count overflow")?;
        // Submission completed successfully before resolving shared sample data.
        let bytes = unsafe { self.samples.resolveCounterRange(NSRange::new(0, count)) }
            .ok_or("Metal timestamp results are unavailable")?
            .to_vec();
        if bytes.len()
            != count
                .checked_mul(8)
                .ok_or("Metal timestamp byte count overflow")?
        {
            return Err("incomplete Metal timestamp results".into());
        }
        let ns = |ticks: u64| -> Result<u64, String> {
            let numerator = u128::from(ticks)
                .checked_mul(u128::from(cpu_delta))
                .and_then(|n| n.checked_add(u128::from(gpu_delta / 2)))
                .ok_or("Metal timestamp conversion overflow")?;
            u64::try_from(numerator / u128::from(gpu_delta))
                .map_err(|_| "Metal timestamp exceeds u64 nanoseconds".into())
        };
        self.dispatches
            .into_iter()
            .enumerate()
            .map(|(i, (invocation, launch, kernel))| {
                let start = u64::from_ne_bytes(bytes[i * 16..i * 16 + 8].try_into().unwrap());
                let end = u64::from_ne_bytes(bytes[i * 16 + 8..i * 16 + 16].try_into().unwrap());
                if start == u64::MAX || end == u64::MAX || end < start || start < self.start.1 {
                    return Err("invalid Metal dispatch timestamps".into());
                }
                let started_ns = ns(start - self.start.1)?;
                let elapsed_ns = ns(end - start)?;
                started_ns
                    .checked_add(elapsed_ns)
                    .ok_or("Metal timestamp endpoint overflow")?;
                Ok(DispatchObservation {
                    invocation,
                    launch,
                    kernel,
                    started_ns,
                    elapsed_ns,
                })
            })
            .collect()
    }
}
