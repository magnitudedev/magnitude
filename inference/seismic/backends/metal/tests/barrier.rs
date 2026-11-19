//! What ordering dependent launches costs on the device, by how a compute
//! encoder orders them: a serial encoder, or a concurrent one with a
//! barrier (buffers scope, or the resources named) before each launch.
//! The direct route's encoding (`direct.rs`, `runtime/src/native/hazard.rs`)
//! rests on these numbers. Run explicitly:
//! `cargo test -p seismic-metal --release --test barrier -- --ignored --nocapture`.
//!
//! M4 Pro, 2026-10-23, µs a launch at 64 KiB / 1 MiB / 8 MiB:
//!
//! | ordering | dependent chain | independent pairs |
//! |---|---|---|
//! | serial encoder | 2.17 / 10.6 / 34.0 | 1.62 / 8.55 / 46.3 |
//! | concurrent, scope barrier | 2.70 / 5.63 / 36.0 | 1.45 / 5.10 / 46.6 |
//! | concurrent, resource barrier | 3.09 / 5.48 / 36.1 | |
//! | concurrent encoder per group | 2.35 / 4.99 / 34.0 | 1.13 / 4.27 / 45.3 |
//! | the same, fenced | 9.40 / 10.3 / 39.9 | 5.03 / 6.14 / 50.3 |
//!
//! Each encoder declaring one order buffer written (the direct route's
//! form), from a second run on a busier host (serial 4.32 / 4.85 / 33.6 and
//! 1.07 / 4.89 / 45.6 there): chain 4.24 / 4.96 / 33.8, pairs 0.78 / 4.14 /
//! 45.2. Independent launches so chained run in order (2.60 ms for 400 at
//! 64 KiB against 0.26 ms unchained).
#![cfg(target_os = "macos")]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLCommonCounterSetTimestamp, MTLComputeCommandEncoder, MTLComputePassDescriptor, MTLCounterSet,
    MTLComputePipelineState, MTLCounterDontSample, MTLCounterSampleBuffer,
    MTLCounterSampleBufferDescriptor, MTLCreateSystemDefaultDevice, MTLDevice, MTLDispatchType,
    MTLFence, MTLLibrary, MTLResource, MTLResourceOptions, MTLResourceUsage, MTLSize,
    MTLStorageMode,
};
use seismic_metal::host_seconds;
use std::ptr::NonNull;

const SOURCE: &str = "#include <metal_stdlib>\nusing namespace metal;\n\
    kernel void step(device const float4* x [[buffer(0)]], device float4* y [[buffer(1)]], \
    uint i [[thread_position_in_grid]]) { y[i] = x[i] * 0.5f + 1.0f; }\n";

#[derive(Clone, Copy, Debug)]
enum Order {
    Serial,
    /// Concurrent, a buffers-scope barrier before every launch but the first.
    Scope,
    /// Concurrent, a barrier naming the launch's two buffers.
    Resources,
    /// Concurrent, every launch independent (its own pair of buffers), no
    /// barrier: the floor.
    Independent,
    /// A concurrent encoder per launch, ordered by Metal's hazard tracking.
    Encoders,
    /// A concurrent encoder per launch, each waiting on a fence the previous
    /// one updates at its end.
    FencedEncoders,
    /// A concurrent encoder per launch, each declaring one order buffer
    /// written (`useResource:usage:`), over the dependent chain.
    Chained,
    /// The same over independent launches: they run in order only if the
    /// declared buffer orders each encoder after the previous one.
    ChainedIndependent,
}

/// What orders an encoder after the one before it.
#[derive(Clone, Copy, PartialEq)]
enum Link {
    /// Only the buffers its dispatches bind (Metal's hazard tracking).
    Tracked,
    Fenced,
    /// A buffer every encoder declares written.
    Chained,
}

struct Bench {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// The first encoder's start (sample 0) and the last one's end (1):
    /// the device time of a run, from GPU timestamp samples.
    samples: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    buffers: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    fence: Retained<ProtocolObject<dyn MTLFence>>,
    order: Retained<ProtocolObject<dyn MTLBuffer>>,
}

type Encoder = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;

/// Which of a run's samples an encoder takes: the run's start when it is
/// the run's first, its end when its last.
#[derive(Clone, Copy)]
struct Ends {
    first: bool,
    last: bool,
}

impl Bench {
    /// An encoder of `dispatch` type, declaring the order buffer when
    /// `link` chains, taking the run's samples `ends` names: only those, so
    /// the samples add nothing to the encoder boundaries measured.
    fn open(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        link: Link,
        dispatch: MTLDispatchType,
        ends: Ends,
    ) -> Encoder {
        let pass = MTLComputePassDescriptor::computePassDescriptor();
        pass.setDispatchType(dispatch);
        // SAFETY: index 0 always exists; both samples lie in the buffer.
        unsafe {
            let attachment = pass.sampleBufferAttachments().objectAtIndexedSubscript(0);
            attachment.setSampleBuffer(Some(&self.samples));
            attachment.setStartOfEncoderSampleIndex(if ends.first { 0 } else { MTLCounterDontSample });
            attachment.setEndOfEncoderSampleIndex(if ends.last { 1 } else { MTLCounterDontSample });
        }
        let encoder = command.computeCommandEncoderWithDescriptor(&pass).unwrap();
        if link == Link::Chained {
            encoder.useResource_usage(
                ProtocolObject::from_ref(&*self.order),
                MTLResourceUsage::Read | MTLResourceUsage::Write,
            );
        }
        encoder.setComputePipelineState(&self.pipeline);
        encoder
    }

    /// End `encoder` and open the next concurrent one after it by `link`,
    /// the run's last when `last`.
    fn next(&self, command: &ProtocolObject<dyn MTLCommandBuffer>, encoder: &mut Encoder, link: Link, last: bool) {
        if link == Link::Fenced {
            encoder.updateFence(&self.fence);
        }
        encoder.endEncoding();
        *encoder = self.open(command, link, MTLDispatchType::Concurrent, Ends { first: false, last });
        if link == Link::Fenced {
            encoder.waitForFence(&self.fence);
        }
    }
}

impl Bench {
    fn new(bytes: usize, launches: usize) -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("a Metal device");
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), None)
            .unwrap();
        let function = library.newFunctionWithName(&NSString::from_str("step")).unwrap();
        let pipeline = device.newComputePipelineStateWithFunction_error(&function).unwrap();
        let buffers = (0..2 * launches)
            .map(|_| {
                device
                    .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModePrivate)
                    .unwrap()
            })
            .collect();
        let counters = device
            .counterSets()
            .and_then(|sets| {
                // SAFETY: a static Metal string constant.
                let wanted = unsafe { MTLCommonCounterSetTimestamp };
                sets.iter().find(|set| &*set.name() == wanted)
            })
            .expect("a timestamp counter set");
        let descriptor = MTLCounterSampleBufferDescriptor::new();
        descriptor.setCounterSet(Some(&counters));
        descriptor.setStorageMode(MTLStorageMode::Shared);
        // SAFETY: Metal validates the count when forming the buffer.
        unsafe { descriptor.setSampleCount(2) };
        let samples = device.newCounterSampleBufferWithDescriptor_error(&descriptor).unwrap();
        Self {
            queue: device.newCommandQueue().unwrap(),
            samples,
            pipeline,
            buffers,
            fence: device.newFence().unwrap(),
            order: device
                .newBufferWithLength_options(16, MTLResourceOptions::StorageModePrivate)
                .unwrap(),
            device,
        }
    }

    /// The GPU and host clocks together.
    fn clocks(&self) -> (u64, f64) {
        let (mut cpu, mut gpu) = (0u64, 0u64);
        let before = host_seconds();
        // SAFETY: both pointers are valid for one write.
        unsafe { self.device.sampleTimestamps_gpuTimestamp(NonNull::from(&mut cpu), NonNull::from(&mut gpu)) };
        (gpu, (before + host_seconds()) / 2.0)
    }

    /// Commit `command`, wait for it, and return its run's device seconds:
    /// from its first encoder's start sample to its last one's end.
    fn run(&self, command: &ProtocolObject<dyn MTLCommandBuffer>) -> f64 {
        let early = self.clocks();
        command.commit();
        command.waitUntilCompleted();
        let late = self.clocks();
        // SAFETY: both samples lie in the buffer; the device wrote them.
        let data = unsafe { self.samples.resolveCounterRange(NSRange::new(0, 2)) }.unwrap().to_vec();
        let tick = |at: usize| u64::from_le_bytes(data[8 * at..8 * at + 8].try_into().unwrap());
        let (start, end) = (tick(0), tick(1));
        assert!(start != u64::MAX && end != u64::MAX && start <= end, "Metal dropped a run's sample");
        (end - start) as f64 * (late.1 - early.1) / (late.0 - early.0) as f64
    }

    /// Device seconds of `launches` launches of `bytes` each, ordered by
    /// `order`: a chain, each launch reading the previous one's output.
    fn seconds(&self, order: Order, bytes: usize, launches: usize) -> f64 {
        let command = self.queue.commandBuffer().unwrap();
        // Whether each launch has an encoder of its own.
        let apart = matches!(
            order,
            Order::Encoders | Order::FencedEncoders | Order::Chained | Order::ChainedIndependent
        );
        let ends = Ends { first: true, last: !apart || launches == 1 };
        let mut encoder = match order {
            Order::Serial => self.open(&command, Link::Tracked, MTLDispatchType::Serial, ends),
            Order::Chained | Order::ChainedIndependent => {
                self.open(&command, Link::Chained, MTLDispatchType::Concurrent, ends)
            }
            _ => self.open(&command, Link::Tracked, MTLDispatchType::Concurrent, ends),
        };
        for launch in 0..launches {
            let (input, output) = match order {
                Order::Independent | Order::ChainedIndependent => {
                    (&self.buffers[2 * launch], &self.buffers[2 * launch + 1])
                }
                _ => (&self.buffers[launch % 2], &self.buffers[(launch + 1) % 2]),
            };
            if launch != 0 {
                match order {
                    Order::Scope => encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers),
                    Order::Resources => {
                        let resources: [NonNull<ProtocolObject<dyn MTLResource>>; 2] = [
                            NonNull::from(ProtocolObject::from_ref(&**input)),
                            NonNull::from(ProtocolObject::from_ref(&**output)),
                        ];
                        // SAFETY: two live resources.
                        unsafe {
                            encoder.memoryBarrierWithResources_count(
                                NonNull::from(&resources[0]),
                                2,
                            )
                        }
                    }
                    Order::Encoders => self.next(&command, &mut encoder, Link::Tracked, launch + 1 == launches),
                    Order::FencedEncoders => self.next(&command, &mut encoder, Link::Fenced, launch + 1 == launches),
                    Order::Chained | Order::ChainedIndependent => {
                        self.next(&command, &mut encoder, Link::Chained, launch + 1 == launches)
                    }
                    Order::Serial | Order::Independent => {}
                }
            }
            // SAFETY: both buffers hold `bytes`, which the grid covers.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
            }
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: bytes / 16,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }
        encoder.endEncoding();
        self.run(&command)
    }
}

/// How a chain of pairs is encoded: each pair's two launches are
/// independent, and each depends on both launches of the previous pair.
#[derive(Clone, Copy, Debug)]
enum Pairs {
    /// One serial encoder.
    Serial,
    /// One concurrent encoder, a buffers-scope barrier before each pair.
    Barriers,
    /// A concurrent encoder per pair, ordered by Metal's hazard tracking.
    Encoders,
    /// A concurrent encoder per pair, each fenced after the previous one.
    FencedEncoders,
    /// A concurrent encoder per pair, each declaring the order buffer.
    Chained,
}

impl Bench {
    fn pairs(&self, encoding: Pairs, bytes: usize, pairs: usize) -> f64 {
        let command = self.queue.commandBuffer().unwrap();
        // Whether each pair has an encoder of its own.
        let apart = matches!(encoding, Pairs::Encoders | Pairs::FencedEncoders | Pairs::Chained);
        let ends = Ends { first: true, last: !apart || pairs == 1 };
        let mut encoder = match encoding {
            Pairs::Serial => self.open(&command, Link::Tracked, MTLDispatchType::Serial, ends),
            Pairs::Chained => self.open(&command, Link::Chained, MTLDispatchType::Concurrent, ends),
            _ => self.open(&command, Link::Tracked, MTLDispatchType::Concurrent, ends),
        };
        for pair in 0..pairs {
            // Pair p reads buffers 2(p mod 2), +1 and writes the other two.
            let read = 2 * (pair % 2);
            let write = 2 * ((pair + 1) % 2);
            if pair != 0 {
                match encoding {
                    Pairs::Barriers => encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers),
                    Pairs::Encoders => self.next(&command, &mut encoder, Link::Tracked, pair + 1 == pairs),
                    Pairs::FencedEncoders => self.next(&command, &mut encoder, Link::Fenced, pair + 1 == pairs),
                    Pairs::Chained => self.next(&command, &mut encoder, Link::Chained, pair + 1 == pairs),
                    Pairs::Serial => {}
                }
            }
            for half in 0..2 {
                // SAFETY: both buffers hold `bytes`, which the grid covers.
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&self.buffers[read + half]), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&self.buffers[write + half]), 0, 1);
                }
                encoder.dispatchThreads_threadsPerThreadgroup(
                    MTLSize {
                        width: bytes / 16,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
            }
        }
        encoder.endEncoding();
        self.run(&command)
    }
}

#[test]
#[ignore = "device micro-benchmark; run explicitly with --ignored --nocapture"]
fn ordering_pairs_of_independent_launches() {
    const PAIRS: usize = 200;
    const REPEATS: usize = 31;
    for bytes in [64 << 10, 1 << 20, 8 << 20] {
        let bench = Bench::new(bytes, 2);
        for encoding in [
            Pairs::Serial,
            Pairs::Barriers,
            Pairs::Encoders,
            Pairs::FencedEncoders,
            Pairs::Chained,
        ] {
            bench.pairs(encoding, bytes, PAIRS);
            let mut samples = (0..REPEATS)
                .map(|_| bench.pairs(encoding, bytes, PAIRS))
                .collect::<Vec<_>>();
            samples.sort_by(f64::total_cmp);
            let median = samples[REPEATS / 2];
            println!(
                "MEASURE {:>5} KiB × {PAIRS} pairs {encoding:?}: {:.3} ms, {:.3} µs a launch",
                bytes >> 10,
                median * 1e3,
                median * 1e6 / (2 * PAIRS) as f64
            );
        }
    }
}

#[test]
#[ignore = "device micro-benchmark; run explicitly with --ignored --nocapture"]
fn ordering_dependent_launches() {
    const LAUNCHES: usize = 400;
    const REPEATS: usize = 31;
    // A decode launch's bytes (a 4B's ~13.6 ms step over ~400 launches is
    // ~34 µs each), and a small one.
    for bytes in [64 << 10, 1 << 20, 8 << 20] {
        let bench = Bench::new(bytes, LAUNCHES);
        for order in [
            Order::Serial,
            Order::Scope,
            Order::Resources,
            Order::Independent,
            Order::Encoders,
            Order::FencedEncoders,
            Order::Chained,
            Order::ChainedIndependent,
        ] {
            bench.seconds(order, bytes, LAUNCHES);
            let mut samples = (0..REPEATS)
                .map(|_| bench.seconds(order, bytes, LAUNCHES))
                .collect::<Vec<_>>();
            samples.sort_by(f64::total_cmp);
            let median = samples[REPEATS / 2];
            println!(
                "MEASURE {:>5} KiB × {LAUNCHES} {order:?}: {:.3} ms, {:.3} µs a launch",
                bytes >> 10,
                median * 1e3,
                median * 1e6 / LAUNCHES as f64
            );
        }
    }
}
