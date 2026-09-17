//! Metal runtime: device, compiled pipelines, buffers, submission, timing.

use crate::msl::{Emitted, Launch};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLBarrierScope, MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};
use std::ptr::NonNull;

pub struct Device {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

pub struct Buffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub len: usize,
}

pub struct Pipeline {
    states: Vec<(Launch, Retained<ProtocolObject<dyn MTLComputePipelineState>>)>,
    pub emitted: Emitted,
    /// Buffers the realization needs and the caller does not supply, allocated at compile.
    scratch: Vec<Buffer>,
}

#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub unified_memory: bool,
    pub max_threads_per_threadgroup: u64,
    pub max_threadgroup_bytes: u64,
    /// GPU cores, from the device's own count of them.
    pub cores: u64,
    pub recommended_working_set_bytes: u64,
}

impl Device {
    pub fn open() -> Result<Device, String> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let queue = device.newCommandQueue().ok_or("could not create a command queue")?;
        Ok(Device { device, queue })
    }

    pub fn info(&self) -> DeviceInfo {
        let size = self.device.maxThreadsPerThreadgroup();
        DeviceInfo {
            name: self.device.name().to_string(),
            unified_memory: self.device.hasUnifiedMemory(),
            max_threads_per_threadgroup: size.width as u64,
            max_threadgroup_bytes: self.device.maxThreadgroupMemoryLength() as u64,
            // Metal does not expose a core count, so it is derived from the working set,
            // which scales with the GPU's size on Apple silicon.
            cores: (self.device.recommendedMaxWorkingSetSize() / (1 << 30)).clamp(8, 128),
            recommended_working_set_bytes: self.device.recommendedMaxWorkingSetSize(),
        }
    }

    pub fn buffer(&self, len: usize) -> Result<Buffer, String> {
        let len = len.max(4);
        let buffer = self.device.newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared).ok_or("buffer allocation failed")?;
        Ok(Buffer { buffer, len })
    }

    pub fn buffer_from(&self, bytes: &[u8]) -> Result<Buffer, String> {
        let b = self.buffer(bytes.len())?;
        b.write(bytes);
        Ok(b)
    }

    pub fn compile(&self, emitted: Emitted) -> Result<Pipeline, String> {
        let source = NSString::from_str(&emitted.source);
        let library = self.device.newLibraryWithSource_options_error(&source, None).map_err(|e| format!("Metal compile failed: {}", e.localizedDescription()))?;
        let mut states = Vec::new();
        for launch in &emitted.launches {
            let name = NSString::from_str(&launch.kernel);
            let function = library.newFunctionWithName(&name).ok_or_else(|| format!("kernel `{}` not found in compiled library", launch.kernel))?;
            let state = self.device.newComputePipelineStateWithFunction_error(&function).map_err(|e| format!("pipeline creation failed: {}", e.localizedDescription()))?;
            states.push((launch.clone(), state));
        }
        let scratch = emitted.scratch.iter().map(|n| self.buffer(*n)).collect::<Result<Vec<_>, _>>()?;
        Ok(Pipeline { states, emitted, scratch })
    }

    /// Submit every launch of the pipeline `repeat` times, in order, in one command buffer, and
    /// wait. Returns GPU time in seconds for the whole command buffer.
    pub fn run(&self, pipeline: &Pipeline, buffers: &[&Buffer], scalars: &[u8], repeat: usize) -> Result<f64, String> {
        let command = self.queue.commandBuffer().ok_or("could not create a command buffer")?;
        for (launch, state) in pipeline.states.iter().cycle().take(pipeline.states.len() * repeat) {
            let encoder = command.computeCommandEncoder().ok_or("could not create a compute encoder")?;
            if launch.after_barrier {
                encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
            }
            encoder.setComputePipelineState(state);
            for (i, b) in buffers.iter().enumerate() {
                unsafe { encoder.setBuffer_offset_atIndex(Some(&b.buffer), 0, i) };
            }
            for (i, b) in pipeline.scratch.iter().enumerate() {
                unsafe { encoder.setBuffer_offset_atIndex(Some(&b.buffer), 0, buffers.len() + i) };
            }
            let scalar_slot = buffers.len() + pipeline.scratch.len();
            if !scalars.is_empty() {
                unsafe { encoder.setBytes_length_atIndex(NonNull::new(scalars.as_ptr() as *mut _).unwrap(), scalars.len(), scalar_slot) };
            }
            let grid = MTLSize { width: launch.threadgroups as usize, height: 1, depth: 1 };
            let group = MTLSize { width: launch.threads_per_threadgroup as usize, height: 1, depth: 1 };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, group);
            encoder.endEncoding();
        }
        command.commit();
        command.waitUntilCompleted();
        if let Some(e) = command.error() {
            return Err(format!("command buffer failed: {}", e.localizedDescription()));
        }
        Ok(command.GPUEndTime() - command.GPUStartTime())
    }
}

impl Device {
    pub(crate) fn queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.queue
    }
}

impl Pipeline {
    pub(crate) fn scratch_buffers(&self) -> &[Buffer] {
        &self.scratch
    }

    pub(crate) fn states(&self) -> &[(Launch, Retained<ProtocolObject<dyn MTLComputePipelineState>>)] {
        &self.states
    }
}

impl Buffer {
    pub(crate) fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }

    /// Fill from bytes at an offset.
    pub fn write_at(&self, offset: usize, bytes: &[u8]) {
        assert!(offset + bytes.len() <= self.len);
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), (self.buffer.contents().as_ptr() as *mut u8).add(offset), bytes.len()) };
    }

    pub fn write(&self, bytes: &[u8]) {
        assert!(bytes.len() <= self.len);
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.buffer.contents().as_ptr() as *mut u8, bytes.len()) };
    }

    pub fn read(&self, len: usize) -> Vec<u8> {
        let mut out = vec![0u8; len];
        unsafe { std::ptr::copy_nonoverlapping(self.buffer.contents().as_ptr() as *const u8, out.as_mut_ptr(), len) };
        out
    }
}

/// Achievable device bandwidth: a streaming read of `bytes` through a reduction kernel, in GB/s.
pub fn calibrate_bandwidth(device: &Device, bytes: usize) -> Result<f64, String> {
    let n = bytes / 4;
    let source = r#"
#include <metal_stdlib>
using namespace metal;
kernel void stream_read(device const float4* x [[buffer(0)]], device float* out [[buffer(1)]],
                        constant uint& n4 [[buffer(2)]],
                        uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint tg [[threadgroup_position_in_grid]]) {
  float acc = 0.0f;
  for (uint i = gid; i < n4; i += 262144u * 4u) { float4 v = x[i]; acc += v.x + v.y + v.z + v.w; }
  acc = simd_sum(acc);
  if (lane == 0) out[tg * 8 + sg] = acc;
}
"#;
    let emitted = Emitted {
        scratch: Vec::new(),
        source: source.to_string(),
        launches: vec![Launch { kernel: "stream_read".into(), threadgroups: 4096, threads_per_threadgroup: 256, after_barrier: false }],
        buffers: Vec::new(),
        scalars: Vec::new(),
    };
    let pipeline = device.compile(emitted)?;
    let input = device.buffer(n * 4)?;
    let out = device.buffer(4096 * 8 * 4)?;
    let n4 = (n / 4) as u32;
    let scalars = n4.to_le_bytes().to_vec();
    let mut best = f64::MAX;
    for _ in 0..8 {
        let t = device.run(&pipeline, &[&input, &out], &scalars, 4)?;
        best = best.min(t / 4.0);
    }
    Ok(n as f64 * 4.0 / best / 1e9)
}
