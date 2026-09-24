//! Compilation and execution of explicitly authored top-level Metal
//! implementations. This path deliberately consumes source plus a closed ABI;
//! it does not construct compiler kernels, plans, schedules, or portfolios.

use crate::{MetalBuffer, MetalDevice};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize,
};
use seismic_compiler::errors::ExecutionError;
use seismic_target::NativeCompilationError;

/// Metal's `setBytes` limit, which bounds a direct entry's argument words.
pub const DIRECT_WORD_BYTES_LIMIT: usize = 4096;

/// One kernel function of an authored Metal library, compiled into a
/// pipeline.
pub struct DirectPipeline {
    state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

// Metal pipeline states are immutable and documented as thread-safe.
unsafe impl Send for DirectPipeline {}
unsafe impl Sync for DirectPipeline {}

impl DirectPipeline {
    /// Compile `source` once and form a pipeline for each named kernel, in
    /// the given order.
    pub fn compile_all(
        device: &MetalDevice,
        source: &str,
        kernels: &[&str],
    ) -> Result<Vec<Self>, NativeCompilationError> {
        let source = NSString::from_str(source);
        let options = objc2_metal::MTLCompileOptions::new();
        #[allow(deprecated)]
        options.setFastMathEnabled(false);
        options.setMathMode(objc2_metal::MTLMathMode::Safe);
        options.setMathFloatingPointFunctions(objc2_metal::MTLMathFloatingPointFunctions::Precise);
        let library = device
            .handle()
            .raw()
            .newLibraryWithSource_options_error(&source, Some(&options))
            .map_err(|error| NativeCompilationError::ToolchainFailure(error.to_string()))?;
        kernels
            .iter()
            .map(|kernel| {
                let name = NSString::from_str(kernel);
                let function = library.newFunctionWithName(&name).ok_or_else(|| {
                    NativeCompilationError::MalformedToolchainOutput(format!(
                        "Metal library does not define kernel `{kernel}`"
                    ))
                })?;
                let state = device
                    .handle()
                    .raw()
                    .newComputePipelineStateWithFunction_error(&function)
                    .map_err(|error| NativeCompilationError::ToolchainFailure(error.to_string()))?;
                Ok(Self { state })
            })
            .collect()
    }

    pub fn max_threads_per_threadgroup(&self) -> u64 {
        self.state.maxTotalThreadsPerThreadgroup() as u64
    }

    /// Threadgroup memory the compiled function declares statically.
    pub fn static_threadgroup_bytes(&self) -> u64 {
        self.state.staticThreadgroupMemoryLength() as u64
    }
}

/// One direct launch as encoded.
pub struct DirectLaunch<'a> {
    pub pipeline: &'a DirectPipeline,
    pub buffers: &'a [(&'a MetalBuffer, u64)],
    pub words: &'a [u8],
    /// Scalar-result slots, bound after the words.
    pub scalar_results: (&'a MetalBuffer, u64),
    pub threadgroups: [u64; 3],
    pub threads_per_threadgroup: [u64; 3],
    pub threadgroup_bytes: u64,
}

/// Ordered direct launches encoded into one serial compute encoder of one
/// command buffer. The serial encoder orders every launch after the previous
/// one's writes.
pub struct DirectBatch {
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
}

impl DirectBatch {
    pub fn new(device: &MetalDevice) -> Result<Self, ExecutionError> {
        let command = device.queue().commandBuffer().ok_or_else(|| {
            ExecutionError::SubmissionFailed(
                "Metal could not create a native command buffer".into(),
            )
        })?;
        let encoder = command.computeCommandEncoder().ok_or_else(|| {
            ExecutionError::SubmissionFailed(
                "Metal could not create a native compute encoder".into(),
            )
        })?;
        Ok(Self { command, encoder })
    }

    pub fn encode(&mut self, launch: &DirectLaunch<'_>) -> Result<(), ExecutionError> {
        if launch.threadgroups.contains(&0) || launch.threads_per_threadgroup.contains(&0) {
            return Ok(());
        }
        let threads = launch
            .threads_per_threadgroup
            .iter()
            .try_fold(1u64, |product, value| product.checked_mul(*value))
            .ok_or_else(|| {
                ExecutionError::SubmissionFailed("native threadgroup size overflowed".into())
            })?;
        if threads > launch.pipeline.max_threads_per_threadgroup() {
            return Err(ExecutionError::SubmissionFailed(format!(
                "native launch requests {threads} threads per threadgroup, but the pipeline allows {}",
                launch.pipeline.max_threads_per_threadgroup()
            )));
        }
        if launch.words.len() > DIRECT_WORD_BYTES_LIMIT {
            return Err(ExecutionError::SubmissionFailed(
                "native ABI words exceed Metal setBytes limit".into(),
            ));
        }
        let encoder = &self.encoder;
        encoder.setComputePipelineState(&launch.pipeline.state);
        for (index, (buffer, offset)) in launch.buffers.iter().enumerate() {
            unsafe { encoder.setBuffer_offset_atIndex(Some(buffer.raw()), *offset as usize, index) }
        }
        let zero = 0u8;
        let words = if launch.words.is_empty() {
            std::slice::from_ref(&zero)
        } else {
            launch.words
        };
        unsafe {
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::from(&words[0]).cast(),
                words.len(),
                launch.buffers.len(),
            );
            encoder.setBuffer_offset_atIndex(
                Some(launch.scalar_results.0.raw()),
                launch.scalar_results.1 as usize,
                launch.buffers.len() + 1,
            );
        }
        if launch.threadgroup_bytes != 0 {
            // Metal requires threadgroup memory lengths in multiples of 16.
            let bytes = launch.threadgroup_bytes.div_ceil(16) * 16;
            unsafe { encoder.setThreadgroupMemoryLength_atIndex(bytes as usize, 0) };
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: launch.threadgroups[0] as usize,
                height: launch.threadgroups[1] as usize,
                depth: launch.threadgroups[2] as usize,
            },
            MTLSize {
                width: launch.threads_per_threadgroup[0] as usize,
                height: launch.threads_per_threadgroup[1] as usize,
                depth: launch.threads_per_threadgroup[2] as usize,
            },
        );
        Ok(())
    }

    /// Commit without waiting. The queue runs command buffers in commit
    /// order.
    pub fn commit(self) -> DirectSubmission {
        self.encoder.endEncoding();
        self.command.commit();
        DirectSubmission {
            command: self.command,
        }
    }
}

/// A committed direct command buffer.
pub struct DirectSubmission {
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

// Waiting on and querying a committed command buffer is thread-safe.
unsafe impl Send for DirectSubmission {}
unsafe impl Sync for DirectSubmission {}

impl DirectSubmission {
    pub fn is_complete(&self) -> bool {
        matches!(
            self.command.status(),
            MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
        )
    }

    pub fn wait_complete(&self) {
        self.command.waitUntilCompleted();
    }

    /// Wait and report the command buffer's outcome.
    pub fn finish(&self) -> Result<(), ExecutionError> {
        self.command.waitUntilCompleted();
        match self.command.error() {
            Some(error) => Err(ExecutionError::SubmissionFailed(
                error.localizedDescription().to_string(),
            )),
            None => Ok(()),
        }
    }

    /// Device execution time of the completed command buffer.
    pub fn device_seconds(&self) -> f64 {
        self.command.GPUEndTime() - self.command.GPUStartTime()
    }
}
