//! Compilation and execution of an explicitly authored top-level Metal
//! implementation. This path deliberately consumes source plus a closed ABI;
//! it does not construct compiler kernels, plans, schedules, or portfolios.

use crate::{MetalBuffer, MetalDevice};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize,
};
use seismic_compiler::errors::{ExecutionError, NativeCompilationError};

/// A pipeline compiled directly from user-authored Metal source.
pub struct DirectPipeline {
    state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

// Metal pipeline states are immutable and documented as thread-safe.
unsafe impl Send for DirectPipeline {}
unsafe impl Sync for DirectPipeline {}

impl DirectPipeline {
    pub fn compile(
        device: &MetalDevice,
        source: &str,
        entry: &str,
    ) -> Result<Self, NativeCompilationError> {
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
        let name = NSString::from_str(entry);
        let function = library.newFunctionWithName(&name).ok_or_else(|| {
            NativeCompilationError::MalformedToolchainOutput(format!(
                "Metal library does not define kernel `{entry}`"
            ))
        })?;
        let state = device
            .handle()
            .raw()
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| NativeCompilationError::ToolchainFailure(error.to_string()))?;
        Ok(Self { state })
    }

    pub fn dispatch(
        &self,
        device: &MetalDevice,
        buffers: &[(&MetalBuffer, u64)],
        threadgroups: [u64; 3],
        threads_per_threadgroup: [u64; 3],
    ) -> Result<(), ExecutionError> {
        if threadgroups.contains(&0) || threads_per_threadgroup.contains(&0) {
            return Ok(());
        }
        let threads = threads_per_threadgroup
            .iter()
            .try_fold(1u64, |product, value| product.checked_mul(*value))
            .ok_or_else(|| {
                ExecutionError::SubmissionFailed("native threadgroup size overflowed".into())
            })?;
        if threads > self.state.maxTotalThreadsPerThreadgroup() as u64 {
            return Err(ExecutionError::SubmissionFailed(format!(
                "native launch requests {threads} threads per threadgroup, but the pipeline allows {}",
                self.state.maxTotalThreadsPerThreadgroup()
            )));
        }
        let command = device.queue().commandBuffer().ok_or_else(|| {
            ExecutionError::SubmissionFailed("Metal could not create a command buffer".into())
        })?;
        let encoder = command.computeCommandEncoder().ok_or_else(|| {
            ExecutionError::SubmissionFailed("Metal could not create a compute encoder".into())
        })?;
        encoder.setComputePipelineState(&self.state);
        for (index, (buffer, offset)) in buffers.iter().enumerate() {
            unsafe { encoder.setBuffer_offset_atIndex(Some(buffer.raw()), *offset as usize, index) }
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: threadgroups[0] as usize,
                height: threadgroups[1] as usize,
                depth: threadgroups[2] as usize,
            },
            MTLSize {
                width: threads_per_threadgroup[0] as usize,
                height: threads_per_threadgroup[1] as usize,
                depth: threads_per_threadgroup[2] as usize,
            },
        );
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            return Err(ExecutionError::SubmissionFailed(
                error.localizedDescription().to_string(),
            ));
        }
        Ok(())
    }
}
