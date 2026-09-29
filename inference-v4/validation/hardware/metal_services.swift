// Independent device observations, never model-candidate benchmarking.
// These measurements are evidence for a timing contract, not a complete contract.
import Foundation
import Metal
import CryptoKit
import Darwin

struct Observation: Codable {
    let name: String
    let pipelineID: String
    let threads: Int
    let iterations: UInt32
    let operationsPerIteration: Int
    let gpuSeconds: [Double]
    let finite: Bool
    let checksum: Double
    var numerical: NumericalCheck? = nil
}
// Numerical comparisons are evidence, not an automatically admitted tolerance.
struct NumericalSample: Codable {
    let inputBits: UInt32
    let expectedBits: UInt32
    let actualBits: UInt32
}
struct NumericalCheck: Codable {
    let reference: String
    let compared: Int
    let nonfinite: Int
    let unequalBits: Int
    let maxAbsoluteError: Double
    let maxRelativeError: Double
    let maxULPDistance: UInt64
    let samples: [NumericalSample]
}
func bf16(_ value: Float) -> Float {
    let bits = value.bitPattern
    if bits & 0x7f800000 == 0x7f800000 { return value }
    return Float(bitPattern: (bits &+ 0x7fff &+ ((bits >> 16) & 1)) & 0xffff0000)
}
func reference(_ name: String, input: Float, iterations: UInt32, copies: Int) -> Float {
    let a: Float = 1.000001, b: Float = 0.00001
    var x = input
    for _ in 0..<Int(iterations) {
        for _ in 0..<copies {
            switch name {
            case "control": x = a
            case "f32_fma": x = fmaf(x, a, b)
            case "f32_add": x = x + a
            case "f32_mul": x = x * a
            case "f32_div": x = x / a
            case "f32_exp": x = expf(-x)
            case "f32_log": x = logf(x + a)
            case "f32_rsqrt": x = 1 / sqrtf(x + a)
            case "f32_sin": x = sinf(x)
            case "f32_cos": x = cosf(x)
            case "f32_max": x = max(x, a)
            case "bf16_roundtrip": x = bf16(x + b)
            case "f16_roundtrip": x = Float(Float16(x + b))
            default: preconditionFailure("missing numerical reference for \(name)")
            }
        }
    }
    return x
}
func compare(_ values: UnsafeMutablePointer<Float>, threads: Int, expected: [Float]) -> NumericalCheck {
    var nonfinite = 0, unequal = 0
    var absolute = 0.0, relative = 0.0
    var ulp: UInt64 = 0
    func ordered(_ value: Float) -> UInt64 {
        let bits = value.bitPattern
        return UInt64(bits & 0x80000000 == 0 ? bits ^ 0x80000000 : ~bits)
    }
    for i in 0..<threads {
        let actual = values[i], target = expected[i % 31]
        if actual.bitPattern != target.bitPattern { unequal += 1 }
        if !actual.isFinite || !target.isFinite { nonfinite += 1; continue }
        let error = abs(Double(actual) - Double(target))
        absolute = max(absolute, error)
        relative = max(relative, error / max(abs(Double(target)), Double(Float.leastNormalMagnitude)))
        let a = ordered(actual), b = ordered(target)
        ulp = max(ulp, max(a, b) - min(a, b))
    }
    let samples = (0..<min(31, threads)).map { i in NumericalSample(
        inputBits: (Float(i + 1) / 32).bitPattern,
        expectedBits: expected[i].bitPattern, actualBits: values[i].bitPattern) }
    return NumericalCheck(reference: "host Float stepwise arithmetic; Darwin fmaf/expf/logf/sinf/cosf; reciprocal sqrtf; round-to-nearest-even BF16/Float16; v1",
        compared: threads, nonfinite: nonfinite, unequalBits: unequal, maxAbsoluteError: absolute,
        maxRelativeError: relative, maxULPDistance: ulp, samples: samples)
}
struct Report: Codable {
    let device: String
    let registryID: UInt64
    let maxThreadsPerThreadgroup: Int
    let maxThreadgroupMemoryLength: Int
    let observations: [Observation]
    let nativeArchives: [NativeArchive]
    let operatingSystem: String
    let sampling: String
    let limitations: [String]
}
struct NativeArchive: Codable {
    let pipelineID: String
    let sourceSHA256: String
    let archiveSHA256: String
    let archiveBytes: Int
    let threadExecutionWidth: Int
    let maxTotalThreadsPerThreadgroup: Int
    let staticThreadgroupMemoryBytes: Int
}
let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let options = MTLCompileOptions()
options.fastMathEnabled = false
var nativeArchives: [NativeArchive] = []
let archiveDirectory: URL? = try {
    guard let index = CommandLine.arguments.firstIndex(of: "--archive-directory") else { return nil }
    guard index + 1 < CommandLine.arguments.count else { throw NSError(domain: "probe", code: 1, userInfo: [NSLocalizedDescriptionKey: "--archive-directory requires a path"]) }
    let url = URL(fileURLWithPath: CommandLine.arguments[index + 1], isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}()
func digest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
func makePipeline(_ source: String, id: String) throws -> MTLComputePipelineState {
    let library = try device.makeLibrary(source: source, options: options)
    let function = library.makeFunction(name: "measure")!
    guard let directory = archiveDirectory else { return try device.makeComputePipelineState(function: function) }
    let descriptor = MTLComputePipelineDescriptor()
    descriptor.computeFunction = function
    let archive = try device.makeBinaryArchive(descriptor: MTLBinaryArchiveDescriptor())
    descriptor.binaryArchives = [archive]
    try archive.addComputePipelineFunctions(descriptor: descriptor)
    let pipeline = try device.makeComputePipelineState(descriptor: descriptor, options: [], reflection: nil)
    let url = directory.appendingPathComponent(id + ".metalar")
    try archive.serialize(to: url)
    let sourceBytes = Data(source.utf8)
    try sourceBytes.write(to: directory.appendingPathComponent(id + ".metal"))
    let archiveBytes = try Data(contentsOf: url)
    nativeArchives.append(NativeArchive(pipelineID: id, sourceSHA256: digest(sourceBytes), archiveSHA256: digest(archiveBytes), archiveBytes: archiveBytes.count,
        threadExecutionWidth: pipeline.threadExecutionWidth, maxTotalThreadsPerThreadgroup: pipeline.maxTotalThreadsPerThreadgroup, staticThreadgroupMemoryBytes: pipeline.staticThreadgroupMemoryLength))
    return pipeline
}
let declarations = """
#include <metal_stdlib>
using namespace metal;
struct Params { uint iterations; uint count; float a; float b; };
"""
let operations: [(String, String, Int)] = [
    ("control", "x = a;", 0),
    ("f32_fma", "x = fma(x, a, b);", 1),
    ("f32_add", "x = x + a;", 1),
    ("f32_mul", "x = x * a;", 1),
    ("f32_div", "x = x / a;", 1),
    ("f32_exp", "x = exp(-x);", 1),
    ("f32_log", "x = log(x + a);", 1),
    ("f32_rsqrt", "x = rsqrt(x + a);", 1),
    ("f32_sin", "x = sin(x);", 1),
    ("f32_cos", "x = cos(x);", 1),
    ("f32_max", "x = max(x, a);", 1),
    ("bf16_roundtrip", "x = float(bfloat(x + b));", 1),
    ("f16_roundtrip", "x = float(half(x + b));", 1)
]
let heldOut = CommandLine.arguments.contains("--held-out")
let copyCounts = heldOut ? [2, 4] : [1, 8]
var results: [Observation] = []
func execute(_ pipeline: MTLComputePipelineState, input: MTLBuffer, output: MTLBuffer, threads: Int, iterations: UInt32) throws -> Double {
    struct Params { var iterations: UInt32; var count: UInt32; var a: Float; var b: Float }
    var params = Params(iterations: iterations, count: UInt32(threads), a: 1.000001, b: 0.00001)
    let command = queue.makeCommandBuffer()!
    let encoder = command.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(input, offset: 0, index: 0)
    encoder.setBuffer(output, offset: 0, index: 1)
    encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 2)
    encoder.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    if let error = command.error { throw error }
    return command.gpuEndTime - command.gpuStartTime
}
for (name, operation, count) in (CommandLine.arguments.contains("--matrix-only") ? [] : operations) {
    for copies in copyCounts {
    let body = Array(repeating: operation, count: copies).joined(separator: "\n")
    let source = declarations + "\n" + """
    kernel void measure(device const float* input [[buffer(0)]], device float* output [[buffer(1)]], constant Params& p [[buffer(2)]], uint id [[thread_position_in_grid]]) {
      float x = input[id];
      float a = p.a, b = p.b;
      for (uint i = 0; i < p.iterations; ++i) { \(body) }
      output[id] = x;
    }
    """
    let pipelineID = "\(name)-\(copies)"
    let pipeline = try makePipeline(source, id: pipelineID)
    for threads in [32, 65536] {
        let input = device.makeBuffer(length: threads * 4, options: .storageModeShared)!
        let output = device.makeBuffer(length: threads * 4, options: .storageModeShared)!
        let data = input.contents().bindMemory(to: Float.self, capacity: threads)
        for i in 0..<threads { data[i] = Float(i % 31 + 1) / 32 }
        for iterations: UInt32 in (heldOut ? [512, 2048] : [256, 1024]) {
            _ = try execute(pipeline, input: input, output: output, threads: threads, iterations: iterations)
            var times: [Double] = []
            for _ in 0..<3 { times.append(try execute(pipeline, input: input, output: output, threads: threads, iterations: iterations)) }
            let values = output.contents().bindMemory(to: Float.self, capacity: threads)
            var checksum: Double = 0
            var finite = true
            for i in 0..<threads { finite = finite && values[i].isFinite; checksum += Double(values[i]) }
            let expected = (0..<31).map { reference(name, input: Float($0 + 1) / 32, iterations: iterations, copies: copies) }
            results.append(Observation(name: name, pipelineID: pipelineID, threads: threads, iterations: iterations, operationsPerIteration: count * copies, gpuSeconds: times, finite: finite, checksum: checksum,
                numerical: compare(values, threads: threads, expected: expected)))
        }
    }
    }
    FileHandle.standardError.write(Data("measured \(name)\n".utf8))
}
// execute() binds count=threads; use a large logical dispatch for an exact byte count.
for count in (CommandLine.arguments.contains("--matrix-only") ? [] : [1 << 20, 1 << 24]) {
    let input = device.makeBuffer(length: count * 4, options: .storageModeShared)!
    let output = device.makeBuffer(length: count * 4, options: .storageModeShared)!
    memset(input.contents(), 0x3f, count * 4)
    // One output per dispatched lane; no overlapping loop writes.
    let directSource = declarations + "\n" + """
    kernel void measure(device const float* input [[buffer(0)]], device float* output [[buffer(1)]], constant Params& p [[buffer(2)]], uint id [[thread_position_in_grid]]) { output[id] = input[id]; }
    """
    let pipelineID = "device-copy-\(count)"
    let direct = try makePipeline(directSource, id: pipelineID)
    _ = try execute(direct, input: input, output: output, threads: count, iterations: 1)
    var times: [Double] = []
    for _ in 0..<3 { times.append(try execute(direct, input: input, output: output, threads: count, iterations: 1)) }
    let correct = memcmp(input.contents(), output.contents(), count * 4) == 0
    results.append(Observation(name: "device_copy", pipelineID: pipelineID, threads: count, iterations: 1, operationsPerIteration: 2, gpuSeconds: times, finite: correct, checksum: Double(output.contents().bindMemory(to: Float.self, capacity: count)[0])))
}

// Independent collective primitive probes. All lanes participate; the uniform
// matrix data produces an exact integer result, checked after every case.
for (dtype, fragment, bytes) in [("float", "simdgroup_float8x8", 4), ("half", "simdgroup_half8x8", 2), ("bfloat", "simdgroup_bfloat8x8", 2)] {
    for copies in copyCounts {
        let operation = Array(repeating: "simdgroup_multiply_accumulate(c, a, b, c);", count: copies).joined(separator: "\n")
        let source = declarations + "\n" + """
        #include <metal_simdgroup_matrix>
        kernel void measure(device const \(dtype)* input [[buffer(0)]], device float* output [[buffer(1)]], constant Params& p [[buffer(2)]], uint id [[thread_position_in_grid]]) {
          uint base = (id / 32) * 64;
          \(fragment) a, b;
          simdgroup_float8x8 c;
          simdgroup_load(a, input, 8);
          simdgroup_load(b, input, 8);
          simdgroup_load(c, output + base, 8);
          for (uint i = 0; i < p.iterations; ++i) { \(operation) }
          simdgroup_store(c, output + base, 8);
        }
        """
        let pipelineID = "matrix-mma-\(dtype)-\(copies)"
        let pipeline = try makePipeline(source, id: pipelineID)
        for threads in [32, 32768] {
            let input = device.makeBuffer(length: 64 * bytes, options: .storageModeShared)!
            if bytes == 4 {
                let values = input.contents().bindMemory(to: Float.self, capacity: 64)
                for i in 0..<64 { values[i] = 0.5 }
            } else {
                let values = input.contents().bindMemory(to: UInt16.self, capacity: 64)
                for i in 0..<64 { values[i] = dtype == "half" ? Float16(0.5).bitPattern : UInt16(Float(0.5).bitPattern >> 16) }
            }
            let elements = threads * 2
            let output = device.makeBuffer(length: elements * 4, options: .storageModeShared)!
            for iterations: UInt32 in (heldOut ? [128, 512] : [64, 256]) {
                var times: [Double] = []
                for sample in 0..<4 {
                    memset(output.contents(), 0, elements * 4)
                    let t = try execute(pipeline, input: input, output: output, threads: threads, iterations: iterations)
                    if sample > 0 { times.append(t) }
                }
                let values = output.contents().bindMemory(to: Float.self, capacity: elements)
                let expected = Float(iterations) * Float(copies) * 2
                var correct = true
                var checksum: Double = 0
                for i in 0..<elements { correct = correct && values[i] == expected; checksum += Double(values[i]) }
                results.append(Observation(name: "matrix_mma_\(dtype)", pipelineID: pipelineID, threads: threads, iterations: iterations, operationsPerIteration: copies, gpuSeconds: times, finite: correct, checksum: checksum))
            }
        }
    }
    FileHandle.standardError.write(Data("measured matrix \(dtype)\n".utf8))
}
let report = Report(device: device.name, registryID: device.registryID, maxThreadsPerThreadgroup: device.maxThreadsPerThreadgroup.width, maxThreadgroupMemoryLength: device.maxThreadgroupMemoryLength, observations: results, nativeArchives: nativeArchives, operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString, sampling: heldOut ? "held-out copies and iterations" : "calibration copies and iterations",
    limitations: ["Whole-kernel GPU timestamps include dispatch overhead.", "Source operations are not native instruction counts; dead-operation and contraction effects must be qualified before mapping.", "Dependent arithmetic chains and streaming copies do not establish occupancy, cache behavior, or mixed-kernel timing.", "Scalar numerical references use host libm and explicit storage rounding; reported errors are not an accepted numerical contract or exhaustive input coverage.", "This report is not a Qwen throughput measurement or a complete hardware model."])
let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
let bytes = try encoder.encode(report)
if CommandLine.arguments.count > 1 { try bytes.write(to: URL(fileURLWithPath: CommandLine.arguments[1])) } else { FileHandle.standardOutput.write(bytes) }
