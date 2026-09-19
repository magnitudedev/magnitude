//! Metal runtime: device, compiled pipelines, buffers, submission, timing.

use crate::msl::{Emitted, Launch};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCompileOptions, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLBarrierScope, MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};
use std::ptr::NonNull;
mod observation;
pub use observation::{DispatchObservation, Observation};

/// Dispatches per committed command buffer of an unprofiled batch.
const COMMIT_DISPATCHES: usize = 64;

pub struct Device {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    identity: std::rc::Rc<()>,
}

#[derive(Clone)]
pub struct Buffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
    offset: usize,
    identity: std::rc::Rc<()>,
}

pub struct Pipeline {
    states: Vec<CompiledLaunch>,
    pub emitted: Emitted,
    /// Buffers the realization needs and the caller does not supply, allocated at compile.
    scratch: Vec<Buffer>,
    identity: std::rc::Rc<()>,
    pub facts: Vec<PipelineFacts>,
}
struct CompiledLaunch {
    index: usize,
    launch: Launch,
    state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}
impl Pipeline {
    /// Selected native dispatches. Absent retained launch slots keep their
    /// identities in the emitted artifact but do not create native pipelines.
    pub fn phase_count(&self) -> usize { self.states.len() }
}

#[derive(Clone, Debug)]
pub struct PipelineFacts {
    /// Original retained launch slot; inactive slots have no native facts.
    pub launch: usize,
    pub kernel: String,
    pub execution_width: u64,
    pub max_threads_per_group: u64,
    pub static_threadgroup_bytes: u64,
}

/// A fully bound invocation retained through synchronous batch completion.
pub struct Invocation<'a> {
    pub pipeline: &'a Pipeline,
    pub buffers: Vec<&'a Buffer>,
    pub scalars: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    pub name: String,
    pub unified_memory: bool,
    pub max_threads_per_threadgroup: u64,
    pub max_threadgroup_bytes: u64,
    /// Unavailable unless obtained from an authoritative device query.
    pub cores: Option<u32>,
    pub recommended_working_set_bytes: u64,
}

impl Device {
    pub fn open() -> Result<Device, String> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let queue = device.newCommandQueue().ok_or("could not create a command queue")?;
        Ok(Device { device, queue, identity: std::rc::Rc::new(()) })
    }

    pub fn info(&self) -> DeviceInfo {
        let size = self.device.maxThreadsPerThreadgroup();
        DeviceInfo {
            name: self.device.name().to_string(),
            unified_memory: self.device.hasUnifiedMemory(),
            max_threads_per_threadgroup: size.width as u64,
            max_threadgroup_bytes: self.device.maxThreadgroupMemoryLength() as u64,
            // Working-set limits do not determine execution-unit count.
            cores: None,
            recommended_working_set_bytes: self.device.recommendedMaxWorkingSetSize(),
        }
    }

    pub fn buffer(&self, len: usize) -> Result<Buffer, String> {
        let buffer = self.device.newBufferWithLength_options(len.max(4), MTLResourceOptions::StorageModeShared).ok_or("buffer allocation failed")?;
        Ok(Buffer { buffer, len, offset: 0, identity: self.identity.clone() })
    }

    pub fn buffer_from(&self, bytes: &[u8]) -> Result<Buffer, String> {
        let b = self.buffer(bytes.len())?;
        b.write(bytes);
        Ok(b)
    }

    pub fn compile(&self, emitted: Emitted) -> Result<Pipeline, String> {
        emitted.scalar_layout()?;
        let scalar_slot = emitted.buffers.len().checked_add(emitted.scratch.len()).ok_or("Metal argument slot overflow")?;
        let status_slot = scalar_slot.checked_add(usize::from(!emitted.scalars.is_empty())).ok_or("Metal argument slot overflow")?;
        if emitted.status_slot.is_some_and(|slot| slot != status_slot) {
            return Err("Metal status binding differs from the selected invocation ABI".into());
        }
        if emitted.scratch.len() != emitted.scratch_bindings.len()
            || emitted.scratch.iter().zip(&emitted.scratch_bindings).any(|(&bytes, binding)| bytes != binding.bytes) {
            return Err("Metal scratch allocations differ from their selected bindings".into());
        }
        if emitted.buffers.iter().chain(&emitted.scratch_bindings).any(|binding| !binding.alignment.is_power_of_two()) {
            return Err("Metal buffer bindings require nonzero power-of-two alignment".into());
        }
        if emitted.alias_pairs.iter().any(|&(left, right, _)| left >= emitted.buffers.len() || right >= emitted.buffers.len()) {
            return Err("Metal alias condition names an absent invocation binding".into());
        }
        for launch in &emitted.launches {
            let absent = launch.bindings.iter().any(|binding| match *binding {
                crate::msl::Binding::Buffer(n) => n >= emitted.buffers.len(),
                crate::msl::Binding::Scratch(n) => n >= emitted.scratch.len(),
                crate::msl::Binding::Scalars => emitted.scalars.is_empty(),
                crate::msl::Binding::Status => false,
            });
            if absent || launch.bindings.len() > crate::msl::MAX_KERNEL_BUFFERS {
                return Err(format!("launch `{}` has an invalid buffer table ({} bindings)", launch.kernel, launch.bindings.len()));
            }
            if let Some(dispatch) = &launch.dispatch {
                if *dispatch != seismic_realization::dispatch::GroupDispatch::new(dispatch.work_items, dispatch.lanes_per_item, dispatch.items_per_group)?
                    || launch.threadgroups != dispatch.groups || launch.threads_per_threadgroup != dispatch.threads_per_group {
                    return Err("Metal launch differs from its selected dispatch geometry".into());
                }
            }
            usize::try_from(launch.threadgroups).map_err(|_| "Metal grid exceeds native dimensions")?;
            usize::try_from(launch.threads_per_threadgroup).map_err(|_| "Metal group exceeds native dimensions")?;
        }
        // Preserve every declared scratch slot, including empty split storage.
        // Removing one would shift the scalar and status arguments of all launches.
        let scratch = emitted.scratch_bindings.iter().map(|binding| {
            let buffer = self.buffer(binding.bytes)?;
            if buffer.allocation_alignment() < binding.alignment as u64 {
                return Err("Metal scratch allocation violates its selected binding alignment".into());
            }
            Ok(buffer)
        }).collect::<Result<Vec<_>, String>>()?;
        if emitted.launches.iter().all(|launch| launch.threadgroups == 0) {
            return Ok(Pipeline { states: Vec::new(), emitted, scratch, identity: self.identity.clone(), facts: Vec::new() });
        }
        let source = NSString::from_str(&emitted.source);
        let options=MTLCompileOptions::new();
        // Keep the macOS 13 API floor. Default Metal fast math may erase
        // publication casts and reassociate explicitly ordered operations.
        #[allow(deprecated)]
        options.setFastMathEnabled(false);
        let library = self.device.newLibraryWithSource_options_error(&source, Some(&options)).map_err(|e| format!("Metal compile failed: {}", e.localizedDescription()))?;
        let mut states = Vec::new();
        let mut facts = Vec::new();
        for (index, launch) in emitted.launches.iter().enumerate() {
            if launch.threadgroups == 0 { continue; }
            let name = NSString::from_str(&launch.kernel);
            let function = library.newFunctionWithName(&name).ok_or_else(|| format!("kernel `{}` not found in compiled library", launch.kernel))?;
            let state = self.device.newComputePipelineStateWithFunction_error(&function).map_err(|e| format!("pipeline creation failed: {}", e.localizedDescription()))?;
            let physical = PipelineFacts {
                launch: index,
                kernel:launch.kernel.clone(),execution_width:state.threadExecutionWidth() as u64,
                max_threads_per_group:state.maxTotalThreadsPerThreadgroup() as u64,
                static_threadgroup_bytes:state.staticThreadgroupMemoryLength() as u64,
            };
            if launch.threads_per_threadgroup==0 || launch.threads_per_threadgroup>physical.max_threads_per_group {
                return Err(format!("{} requests {} threads but native pipeline permits {}",launch.kernel,launch.threads_per_threadgroup,physical.max_threads_per_group));
            }
            if launch.declared_threadgroup_bytes > self.device.maxThreadgroupMemoryLength() as u64
                || physical.static_threadgroup_bytes > self.device.maxThreadgroupMemoryLength() as u64 {
                return Err("native pipeline threadgroup storage exceeds device capacity".into());
            }
            if let Some(dispatch)=&launch.dispatch {
                if physical.execution_width != dispatch.lanes_per_item || launch.threadgroups != dispatch.groups || launch.threads_per_threadgroup != dispatch.threads_per_group {
                    return Err("native pipeline/launch differs from declared subgroup realization".into());
                }
            }
            facts.push(physical);
            states.push(CompiledLaunch { index, launch: launch.clone(), state });
        }
        Ok(Pipeline { states, emitted, scratch, identity:self.identity.clone(), facts })
    }

    /// Submit every launch of the pipeline `repeat` times, in order, in one command buffer, and
    /// wait. Returns GPU time in seconds for the whole command buffer.
    pub fn run(&self, pipeline: &Pipeline, buffers: &[&Buffer], scalars: &[u8], repeat: usize) -> Result<f64, String> {
        self.run_many(&[Invocation {pipeline,buffers:buffers.to_vec(),scalars:scalars.to_vec()}],repeat)
    }

    /// Validate every binding before submission, encode in source order, and
    /// retain one error status through the entire command buffer. Later dispatches
    /// cannot erase an earlier failure. Physical completion precedes return.
    pub fn run_many(&self, invocations: &[Invocation<'_>], repeat: usize) -> Result<f64, String> {
        Ok(self.submit(invocations, repeat, false)?.command_seconds)
    }

    /// Optional qualification: one sampled compute encoder per dispatch. Its
    /// stage interval differs from the uninstrumented command-buffer interval.
    /// Unsupported native counters are reported; no substituted clock is used.
    pub fn profile(
        &self,
        pipeline: &Pipeline,
        buffers: &[&Buffer],
        scalars: &[u8],
    ) -> Result<Observation, String> {
        self.submit(
            &[Invocation {
                pipeline,
                buffers: buffers.to_vec(),
                scalars: scalars.to_vec(),
            }],
            1,
            true,
        )
    }

    fn submit(
        &self,
        invocations: &[Invocation<'_>],
        repeat: usize,
        profile: bool,
    ) -> Result<Observation, String> {
        if repeat == 0 || invocations.is_empty() {
            return Err("Metal batch and repeat count must be nonempty".into());
        }
        let mut dispatch_count = 0usize;
        for invocation in invocations {
            let Invocation {
                pipeline,
                buffers,
                scalars,
            } = invocation;
            self.validate_pipeline(pipeline)?;
            if buffers.len() != pipeline.emitted.buffers.len() {
                return Err("Metal buffer binding count mismatch".into());
            }
            for (buffer, slot) in buffers.iter().zip(&pipeline.emitted.buffers) {
                self.validate_buffer(buffer)?;
                if buffer.allocation_alignment() < slot.alignment as u64 || !buffer.offset.is_multiple_of(slot.alignment) {
                    return Err("Metal resident view violates typed storage alignment".into());
                }
                if buffer.len() < slot.bytes {
                    return Err(format!(
                        "Metal buffer {}.{} has {} bytes; needs {}",
                        slot.parameter,
                        slot.plane,
                        buffer.len(),
                        slot.bytes
                    ));
                }
            }
            for (a, b, exact_allowed) in &pipeline.emitted.alias_pairs {
                let (left, right) = (buffers[*a], buffers[*b]);
                let (left_size, right_size) = (
                    pipeline.emitted.buffers[*a].bytes,
                    pipeline.emitted.buffers[*b].bytes,
                );
                if std::ptr::eq(left.raw(), right.raw())
                    && left_size != 0 && right_size != 0
                    && left.offset < right.offset + right_size
                    && right.offset < left.offset + left_size
                    && !(*exact_allowed && left.offset == right.offset && left_size == right_size)
                {
                    return Err("pointwise partition binding has unsafe overlapping storage".into());
                }
            }
            pipeline.emitted.scalar_layout()?.validate_bytes(scalars)?;
            dispatch_count = dispatch_count
                .checked_add(pipeline.phase_count())
                .ok_or("Metal dispatch count overflow")?;
        }
        let dispatch_count = dispatch_count
            .checked_mul(repeat)
            .ok_or("Metal repetition overflow")?;
        if dispatch_count == 0 {
            return Ok(Observation { command_seconds: 0.0, dispatches: Vec::new() });
        }
        let status = self.buffer_from(&[0; 4])?;
        let mut capture = if profile {
            Some(observation::Capture::new(self, dispatch_count)?)
        } else {
            None
        };
        let open = || -> Result<_, String> {
            let command = self.queue.commandBuffer().ok_or("could not create a command buffer")?;
            let encoder = if profile { None } else { Some(command.computeCommandEncoder().ok_or("could not create a compute encoder")?) };
            Ok((command, encoder))
        };
        // Submission rule: an unprofiled batch is committed in command buffers of
        // `COMMIT_DISPATCHES` dispatches, in source order on the one queue, so the device
        // executes the head of the batch while the host encodes the rest. Completion, error
        // and status checks still cover the whole batch before return.
        let mut committed = Vec::new();
        let (mut command, mut shared_encoder) = open()?;
        let mut encoded = 0usize;
        let mut first = true;
        for _ in 0..repeat {
            for (invocation_index, invocation) in invocations.iter().enumerate() {
                let Invocation {
                    pipeline,
                    buffers,
                    scalars,
                } = invocation;
                for compiled in &pipeline.states {
                    let CompiledLaunch { index: launch_index, launch, state } = compiled;
                    if !profile && encoded == COMMIT_DISPATCHES {
                        if let Some(encoder) = &shared_encoder {
                            encoder.endEncoding();
                        }
                        command.commit();
                        committed.push(command);
                        (command, shared_encoder) = open()?;
                        (encoded, first) = (0, true);
                    }
                    encoded += 1;
                    let encoder = match (&mut capture, &shared_encoder) {
                        (Some(capture), _) => capture.encoder(&command, invocation_index, *launch_index, &launch.kernel)?,
                        (None, Some(encoder)) => encoder.clone(),
                        (None, None) => return Err("Metal submission has no compute encoder".into()),
                    };
                    if !profile && (!first || launch.after_barrier) {
                        encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
                    }
                    first = false;
                    encoder.setComputePipelineState(state);
                    // Each kernel declares only the resources it references; its table maps
                    // local buffer index -> invocation resource (validated by `compile`).
                    for (local, binding) in launch.bindings.iter().enumerate() {
                        match *binding {
                            crate::msl::Binding::Buffer(n) => {
                                let b = buffers.get(n).ok_or("Metal launch binds an absent invocation buffer")?;
                                unsafe { encoder.setBuffer_offset_atIndex(Some(&b.buffer), b.offset, local) };
                            }
                            crate::msl::Binding::Scratch(n) => {
                                let b = pipeline.scratch.get(n).ok_or("Metal launch binds absent scratch storage")?;
                                unsafe { encoder.setBuffer_offset_atIndex(Some(&b.buffer), 0, local) };
                            }
                            crate::msl::Binding::Status => unsafe { encoder.setBuffer_offset_atIndex(Some(&status.buffer), 0, local) },
                            crate::msl::Binding::Scalars => {
                                let bytes = NonNull::new(scalars.as_ptr() as *mut _).ok_or("Metal scalar block has no storage")?;
                                unsafe { encoder.setBytes_length_atIndex(bytes, scalars.len(), local) };
                            }
                        }
                    }
                    let grid = MTLSize { width: launch.threadgroups as usize, height: 1, depth: 1 };
                    let group = MTLSize { width: launch.threads_per_threadgroup as usize, height: 1, depth: 1 };
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, group);
                    if profile {
                        encoder.endEncoding();
                    }
                }
            }
        }
        if let Some(encoder) = shared_encoder {
            encoder.endEncoding();
        }
        command.commit();
        committed.push(command);
        for command in &committed {
            command.waitUntilCompleted();
            if let Some(e) = command.error() {
                return Err(format!("command buffer failed: {}", e.localizedDescription()));
            }
        }
        if status.read(4) != [0; 4] {
            return Err("Metal invocation encountered an out-of-bounds view".into());
        }
        Ok(Observation {
            // First start to last end on the one queue, including any wait for the host.
            command_seconds: committed.last().zip(committed.first()).map_or(0.0, |(last, first)| last.GPUEndTime() - first.GPUStartTime()),
            dispatches: capture
                .map(|capture| capture.finish(self))
                .transpose()?
                .unwrap_or_default(),
        })
    }
}

impl Device {
    pub(crate) fn validate_buffer(&self, buffer: &Buffer) -> Result<(),String> {
        if !std::rc::Rc::ptr_eq(&self.identity,&buffer.identity) { return Err("Metal buffer belongs to a different device owner".into()); }
        Ok(())
    }
    pub(crate) fn validate_pipeline(&self, pipeline: &Pipeline) -> Result<(),String> {
        if !std::rc::Rc::ptr_eq(&self.identity,&pipeline.identity) { return Err("Metal pipeline belongs to a different device owner".into()); }
        Ok(())
    }
}

impl Buffer {
    /// Alignment of this allocation's GPU virtual address, independent of the
    /// CPU mapping and any retained view offset. A missing address proves only
    /// byte alignment; it does not justify an assumed page or SIMD alignment.
    pub fn allocation_alignment(&self) -> u64 {
        let address = self.buffer.gpuAddress();
        if address == 0 { 1 } else { 1u64 << address.trailing_zeros() }
    }
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    pub fn view(&self, range: std::ops::Range<usize>) -> Result<Self,String> {
        if range.start > range.end || range.end > self.len { return Err("Metal buffer view exceeds its parent".into()); }
        Ok(Self { buffer:self.buffer.clone(),len:range.end-range.start,offset:self.offset.checked_add(range.start).ok_or("Metal view offset overflow")?,identity:self.identity.clone() })
    }
    pub(crate) fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }

    /// Fill from bytes at an offset.
    pub fn write_at(&self, offset: usize, bytes: &[u8]) {
        assert!(offset.checked_add(bytes.len()).is_some_and(|end| end <= self.len));
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), (self.buffer.contents().as_ptr() as *mut u8).add(self.offset + offset), bytes.len()) };
    }

    pub fn write(&self, bytes: &[u8]) {
        assert!(bytes.len() <= self.len);
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), (self.buffer.contents().as_ptr() as *mut u8).add(self.offset), bytes.len()) };
    }

    pub fn read(&self, len: usize) -> Vec<u8> {
        assert!(len <= self.len, "Metal host read exceeds buffer capacity");
        let mut out = vec![0u8; len];
        unsafe { std::ptr::copy_nonoverlapping((self.buffer.contents().as_ptr() as *const u8).add(self.offset), out.as_mut_ptr(), len) };
        out
    }
}
