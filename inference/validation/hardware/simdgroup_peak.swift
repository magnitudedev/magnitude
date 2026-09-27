// P4b: Apple GPU simdgroup_matrix 8x8 MMA peak with register-resident operands.
//
// Unlike `simdgroup_throughput.swift` (operands reloaded from threadgroup memory every
// iteration), each simdgroup loads NA A and NB B fragments from device memory once and
// then issues NA*NB MMAs per iteration into NA*NB independent accumulator chains
// acc = a_x * b_y + acc. No memory instruction is inside the loop, so the slope of GPU
// time over the iteration count is the matrix-unit issue rate alone. Every accumulator
// is stored, so no chain is dead; a chain cannot be folded because each step depends on
// the previous accumulator (safe math, no reassociation). TFLOP/s comes from the fitted
// per-iteration slope over several iteration counts.
//
//   swiftc -O simdgroup_peak.swift -o simdgroup-peak
//   ./simdgroup-peak [--threadgroups-per-core 32] [--threads 256] > peak-<host>.json
import Foundation
import Metal
import IOKit

var repetitions = 10
var warmup = 2
var threadgroupsPerCore = 32
var threadsPerThreadgroup = 256
var iterationCounts = [1024, 2048, 4096, 8192]
do {
    var iterator = CommandLine.arguments.dropFirst().makeIterator()
    while let flag = iterator.next() {
        guard let value = iterator.next() else { fatalError("\(flag) requires a value") }
        switch flag {
        case "--repetitions": repetitions = Int(value)!
        case "--threadgroups-per-core": threadgroupsPerCore = Int(value)!
        case "--threads": threadsPerThreadgroup = Int(value)!
        case "--iterations": iterationCounts = value.split(separator: ",").map { Int($0)! }
        default: fatalError("unknown flag \(flag)")
        }
    }
    precondition(iterationCounts.count >= 2, "the slope needs at least two iteration counts")
}

func gpuCoreCount() -> Int {
    let service = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"))
    guard service != 0 else { fatalError("no AGXAccelerator in IORegistry") }
    defer { IOObjectRelease(service) }
    guard let property = IORegistryEntryCreateCFProperty(service, "gpu-core-count" as CFString, kCFAllocatorDefault, 0)?.takeRetainedValue() as? Int else {
        fatalError("AGXAccelerator has no gpu-core-count property")
    }
    return property
}
func hostName() -> String {
    var buffer = [CChar](repeating: 0, count: 256)
    precondition(gethostname(&buffer, buffer.count) == 0)
    return String(cString: buffer)
}
func median(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    return sorted.count % 2 == 1 ? sorted[sorted.count / 2] : (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
}

struct Variant: Codable { let operand: String; let accumulator: String }
struct Shape: Codable { let na: Int; let nb: Int }

func source(_ variant: Variant, _ shape: Shape) -> String {
    let t = variant.operand, acc = variant.accumulator
    let accumulators = shape.na * shape.nb
    var body = ""
    for x in 0..<shape.na { body += "  simdgroup_matrix<\(t), 8, 8> a\(x);\n  simdgroup_load(a\(x), a_src + ((sg + \(x)u) & 15u) * 64u, 8);\n" }
    for y in 0..<shape.nb { body += "  simdgroup_matrix<\(t), 8, 8> b\(y);\n  simdgroup_load(b\(y), b_src + ((sg + \(y)u + 3u) & 15u) * 64u, 8);\n" }
    for j in 0..<accumulators { body += "  simdgroup_matrix<\(acc), 8, 8> c\(j);\n  simdgroup_load(c\(j), out + (global_sg * \(accumulators)u + \(j)u) * 64u, 8);\n" }
    body += "  for (uint it = 0; it < iterations; ++it) {\n"
    for x in 0..<shape.na { for y in 0..<shape.nb { body += "    simdgroup_multiply_accumulate(c\(x * shape.nb + y), a\(x), b\(y), c\(x * shape.nb + y));\n" } }
    body += "  }\n"
    for j in 0..<accumulators { body += "  simdgroup_store(c\(j), out + (global_sg * \(accumulators)u + \(j)u) * 64u, 8);\n" }
    return """
    #include <metal_stdlib>
    #include <metal_simdgroup_matrix>
    using namespace metal;
    kernel void mma(device const \(t)* a_src [[buffer(0)]], device const \(t)* b_src [[buffer(1)]],
                    device \(acc)* out [[buffer(2)]], constant uint& iterations [[buffer(3)]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint sgs [[simdgroups_per_threadgroup]],
                    uint tg [[threadgroup_position_in_grid]]) {
      uint global_sg = tg * sgs + sg;
    \(body)
    }
    """
}

struct Point: Codable { let iterations: Int; let medianSeconds: Double; let tflops: Double }
struct Result: Codable {
    let operand: String
    let accumulator: String
    let shape: Shape
    let threadgroups: Int
    let threadsPerThreadgroup: Int
    let points: [Point]
    let linearityMaxRelativeResidual: Double
    let tflopsFromSlope: Double
}
struct Report: Codable {
    let probe: String
    let host: String
    let device: String
    let gpuCores: Int
    let operatingSystem: String
    let bestPerVariant: [String: Double]
    let results: [Result]
}

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let cores = gpuCoreCount()
let options = MTLCompileOptions()
options.mathMode = .safe
options.languageVersion = .version3_1

func bytes(_ type: String, _ count: Int) -> MTLBuffer {
    // Values in [-1/64, 1/64]: long chains stay finite in every type.
    let buffer = device.makeBuffer(length: count * (type == "float" ? 4 : 2), options: .storageModeShared)!
    let pointer = buffer.contents()
    for i in 0..<count {
        let value = Float((i * 37) % 129 - 64) / 4096
        switch type {
        case "float": pointer.storeBytes(of: value, toByteOffset: i * 4, as: Float.self)
        case "half": pointer.storeBytes(of: Float16(value).bitPattern, toByteOffset: i * 2, as: UInt16.self)
        default: pointer.storeBytes(of: UInt16(value.bitPattern >> 16), toByteOffset: i * 2, as: UInt16.self)
        }
    }
    return buffer
}

func dispatch(_ pipeline: MTLComputePipelineState, _ a: MTLBuffer, _ b: MTLBuffer, _ out: MTLBuffer, threadgroups: Int, iterations: Int) -> Double {
    var count = UInt32(iterations)
    let command = queue.makeCommandBuffer()!
    let encoder = command.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(a, offset: 0, index: 0)
    encoder.setBuffer(b, offset: 0, index: 1)
    encoder.setBuffer(out, offset: 0, index: 2)
    encoder.setBytes(&count, length: 4, index: 3)
    encoder.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: threadsPerThreadgroup, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    if let error = command.error { fatalError("command buffer failed: \(error)") }
    return command.gpuEndTime - command.gpuStartTime
}

let variants = [Variant(operand: "half", accumulator: "float"), Variant(operand: "half", accumulator: "half"),
                Variant(operand: "bfloat", accumulator: "float"), Variant(operand: "bfloat", accumulator: "bfloat"),
                Variant(operand: "float", accumulator: "float")]
let shapes = [Shape(na: 1, nb: 1), Shape(na: 1, nb: 4), Shape(na: 2, nb: 2), Shape(na: 2, nb: 4), Shape(na: 4, nb: 4)]
var results: [Result] = []
for variant in variants {
    let a = bytes(variant.operand, 16 * 64), b = bytes(variant.operand, 16 * 64)
    for shape in shapes {
        let library = try! device.makeLibrary(source: source(variant, shape), options: options)
        let pipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "mma")!)
        precondition(pipeline.maxTotalThreadsPerThreadgroup >= threadsPerThreadgroup,
            "\(variant) \(shape): pipeline admits only \(pipeline.maxTotalThreadsPerThreadgroup) threads")
        let threadgroups = cores * threadgroupsPerCore
        let simdgroups = threadgroups * threadsPerThreadgroup / 32
        let accumulators = shape.na * shape.nb
        let out = bytes(variant.accumulator, simdgroups * accumulators * 64)
        let flopsPerIteration = Double(simdgroups * accumulators * 2 * 512)
        var times = Array(repeating: [Double](), count: iterationCounts.count)
        for sample in 0..<(warmup + repetitions) {
            for (index, iterations) in iterationCounts.enumerated() {
                let t = dispatch(pipeline, a, b, out, threadgroups: threadgroups, iterations: iterations)
                if sample >= warmup { times[index].append(t) }
            }
        }
        let points = zip(iterationCounts, times).map { iterations, times in
            Point(iterations: iterations, medianSeconds: median(times), tflops: flopsPerIteration * Double(iterations) / median(times) / 1e12)
        }
        let xs = points.map { Double($0.iterations) }, ys = points.map { $0.medianSeconds }
        let mx = xs.reduce(0, +) / Double(xs.count), my = ys.reduce(0, +) / Double(ys.count)
        let slope = zip(xs, ys).map { ($0 - mx) * ($1 - my) }.reduce(0, +) / xs.map { ($0 - mx) * ($0 - mx) }.reduce(0, +)
        let intercept = my - slope * mx
        let residual = zip(xs, ys).map { abs($1 - (intercept + slope * $0)) / $1 }.max()!
        let result = Result(operand: variant.operand, accumulator: variant.accumulator, shape: shape, threadgroups: threadgroups,
            threadsPerThreadgroup: threadsPerThreadgroup, points: points, linearityMaxRelativeResidual: residual,
            tflopsFromSlope: flopsPerIteration / slope / 1e12)
        results.append(result)
        FileHandle.standardError.write(Data(String(format: "%@x%@->%@ %dx%d: %.2f TFLOP/s (slope), residual %.3f\n",
            variant.operand, variant.operand, variant.accumulator, shape.na, shape.nb, result.tflopsFromSlope, residual).utf8))
    }
}
var best: [String: Double] = [:]
for result in results {
    let key = "\(result.operand)->\(result.accumulator)"
    best[key] = max(best[key] ?? 0, result.tflopsFromSlope)
}
let report = Report(probe: "simdgroup_peak", host: hostName(), device: device.name, gpuCores: cores,
    operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString, bestPerVariant: best, results: results)
let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
FileHandle.standardOutput.write(try! encoder.encode(report))
FileHandle.standardOutput.write(Data("\n".utf8))
