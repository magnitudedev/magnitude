import Foundation
import Metal

func fail(_ message: String) -> Never {
  fputs(message + "\n", stderr)
  exit(1)
}

guard let device = MTLCreateSystemDefaultDevice() else { fail("No default Metal device") }
let source = #"""
#include <metal_stdlib>
using namespace metal;
kernel void reduce32(device const float *input [[buffer(0)]], device float *output [[buffer(1)]], uint lane [[thread_index_in_simdgroup]]) {
  float total = simd_sum(input[lane]);
  if (lane == 0) output[0] = total;
}
"""#
let library: MTLLibrary
do { library = try device.makeLibrary(source: source, options: nil) }
catch { fail("SIMD reduction library compilation failed: \(error)") }
guard let function = library.makeFunction(name: "reduce32") else { fail("SIMD reduction function is missing") }
let pipeline: MTLComputePipelineState
do { pipeline = try device.makeComputePipelineState(function: function) }
catch { fail("SIMD reduction pipeline creation failed: \(error)") }
let input = (1...32).map(Float.init)
guard let inputBuffer = device.makeBuffer(bytes: input, length: input.count * MemoryLayout<Float>.size),
      let outputBuffer = device.makeBuffer(length: MemoryLayout<Float>.size),
      let queue = device.makeCommandQueue(), let command = queue.makeCommandBuffer(),
      let encoder = command.makeComputeCommandEncoder() else { fail("SIMD reduction command allocation failed") }
encoder.setComputePipelineState(pipeline)
encoder.setBuffer(inputBuffer, offset: 0, index: 0)
encoder.setBuffer(outputBuffer, offset: 0, index: 1)
encoder.dispatchThreads(MTLSize(width: 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
encoder.endEncoding()
command.commit()
command.waitUntilCompleted()
guard command.status == .completed else { fail("SIMD reduction execution failed: \(String(describing: command.error))") }
let result = outputBuffer.contents().bindMemory(to: Float.self, capacity: 1).pointee
guard abs(result - 528) < 0.001 else { fail("SIMD reduction returned \(result), expected 528") }
let report: [String: Any] = [
  "schemaVersion": 1,
  "device": device.name,
  "registryID": String(device.registryID),
  "apple7": device.supportsFamily(.apple7),
  "metal3": device.supportsFamily(.metal3),
  "maxThreadgroupMemory": device.maxThreadgroupMemoryLength,
  "simdReductionExecuted": true,
  "simdReductionResult": result,
]
let data = try! JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
print(String(data: data, encoding: .utf8)!)
