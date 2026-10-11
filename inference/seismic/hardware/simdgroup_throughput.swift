// P4: Apple GPU simdgroup_matrix 8x8 MMA throughput per operand type, with
// operands that cannot be constant-folded or hoisted.
//
// Every threadgroup stages FRAGMENTS distinct 8x8 A and B fragments from device memory
// into threadgroup memory. Iteration `it` of simdgroup `sg` loads A fragments
// (sg + it*NA + x) & mask and B fragments (sg + it*NB + y + 3) & mask, where `mask`
// is a runtime argument, so each iteration multiplies different data. Each of the
// NA*NB accumulators is a data-dependent chain acc = a_x * b_y + acc starting from
// values loaded from device memory, and every accumulator is stored to device memory.
// Folding is excluded by (a) a CPU reference for a small dispatch and (b) timing that
// is linear in the iteration count; TFLOP/s comes from the fitted per-iteration slope.
//
//   swiftc -O simdgroup_throughput.swift -o simdgroup-throughput
//   ./simdgroup-throughput [--threadgroups-per-core 16] [--repetitions 10] > simdgroup-<host>.json
import Foundation
import Metal
import IOKit

let fragments = 16
let threadsPerThreadgroup = 256
var repetitions = 10
var warmup = 2
var threadgroupsPerCore = 32
var iterationCounts = [512, 1024, 2048, 4096]
do {
    var iterator = CommandLine.arguments.dropFirst().makeIterator()
    while let flag = iterator.next() {
        guard let value = iterator.next() else { fatalError("\(flag) requires a value") }
        switch flag {
        case "--repetitions": repetitions = Int(value)!
        case "--warmup": warmup = Int(value)!
        case "--threadgroups-per-core": threadgroupsPerCore = Int(value)!
        case "--iterations": iterationCounts = value.split(separator: ",").map { Int($0)! }
        default: fatalError("unknown flag \(flag)")
        }
    }
    precondition(repetitions >= 10, "at least 10 timed repetitions are required")
    precondition(iterationCounts.count >= 2, "the linearity check needs at least two iteration counts")
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
    let middle = sorted.count / 2
    return sorted.count % 2 == 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2
}

enum Scalar: String, Codable {
    case float, half, bfloat
    var bytes: Int { self == .float ? 4 : 2 }
    func encode(_ value: Float) -> [UInt8] {
        switch self {
        case .float: return withUnsafeBytes(of: value.bitPattern.littleEndian, Array.init)
        case .half: return withUnsafeBytes(of: Float16(value).bitPattern.littleEndian, Array.init)
        case .bfloat:
            // Every probe value is a multiple of 1/64 in [-1, 1], exact in bfloat.
            precondition(value.bitPattern & 0xffff == 0)
            return withUnsafeBytes(of: UInt16(value.bitPattern >> 16).littleEndian, Array.init)
        }
    }
    func decode(_ pointer: UnsafeRawPointer, _ index: Int) -> Float {
        switch self {
        case .float: return pointer.load(fromByteOffset: index * 4, as: Float.self)
        case .half: return Float(pointer.load(fromByteOffset: index * 2, as: Float16.self))
        case .bfloat: return Float(bitPattern: UInt32(pointer.load(fromByteOffset: index * 2, as: UInt16.self)) << 16)
        }
    }
}

struct Shape: Codable { let na: Int; let nb: Int }
struct Variant { let operand: Scalar; let accumulator: Scalar }

func source(_ variant: Variant, _ shape: Shape) -> String {
    let t = variant.operand.rawValue, acc = variant.accumulator.rawValue
    let accumulators = shape.na * shape.nb
    var body = ""
    for j in 0..<accumulators { body += "  simdgroup_matrix<\(acc), 8, 8> c\(j);\n  simdgroup_load(c\(j), out + (global_sg * \(accumulators)u + \(j)u) * 64u, 8);\n" }
    body += "  for (uint it = 0; it < p.iterations; ++it) {\n"
    for x in 0..<shape.na { body += "    simdgroup_matrix<\(t), 8, 8> a\(x);\n    simdgroup_load(a\(x), ta + ((sg + it * \(shape.na)u + \(x)u) & p.mask) * 64u, 8);\n" }
    for y in 0..<shape.nb { body += "    simdgroup_matrix<\(t), 8, 8> b\(y);\n    simdgroup_load(b\(y), tb + ((sg + it * \(shape.nb)u + \(y)u + 3u) & p.mask) * 64u, 8);\n" }
    for x in 0..<shape.na { for y in 0..<shape.nb { let j = x * shape.nb + y; body += "    simdgroup_multiply_accumulate(c\(j), a\(x), b\(y), c\(j));\n" } }
    body += "  }\n"
    for j in 0..<accumulators { body += "  simdgroup_store(c\(j), out + (global_sg * \(accumulators)u + \(j)u) * 64u, 8);\n" }
    return """
    #include <metal_stdlib>
    #include <metal_simdgroup_matrix>
    using namespace metal;
    struct Params { uint iterations; uint mask; };
    kernel void mma(device const \(t)* a_src [[buffer(0)]], device const \(t)* b_src [[buffer(1)]],
                    device \(acc)* out [[buffer(2)]], constant Params& p [[buffer(3)]],
                    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint sgs [[simdgroups_per_threadgroup]], uint tg [[threadgroup_position_in_grid]]) {
      threadgroup \(t) ta[\(fragments * 64)];
      threadgroup \(t) tb[\(fragments * 64)];
      for (uint i = tid; i < \(fragments * 64)u; i += sgs * 32u) { ta[i] = a_src[i]; tb[i] = b_src[i]; }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      uint global_sg = tg * sgs + sg;
    \(body)
    }
    """
}

struct Point: Codable {
    let iterations: Int
    let gpuSeconds: [Double]
    let medianSeconds: Double
    let tflops: Double
}
struct Check: Codable {
    let threadgroups: Int
    let iterations: Int
    let elements: Int
    let unequalBits: Int
    let maxAbsoluteError: Double
    let maxRelativeError: Double
    let reference: String
}
struct Result: Codable {
    let operand: Scalar
    let accumulator: Scalar
    let accumulators: Int
    let shape: Shape
    let operandLoadsPerIteration: Int
    let mmasPerIteration: Int
    let threadgroups: Int
    let simdgroups: Int
    let check: Check
    let points: [Point]
    let slopeSecondsPerIteration: Double
    let interceptSeconds: Double
    let linearityMaxRelativeResidual: Double
    let tflopsFromSlope: Double
}
struct Report: Codable {
    let probe: String
    let backend: String
    let host: String
    let device: String
    let gpuCores: Int
    let operatingSystem: String
    let languageVersion: String
    let threadsPerThreadgroup: Int
    let fragmentsInThreadgroupMemory: Int
    let repetitions: Int
    let warmup: Int
    let definitions: [String: String]
    let bestPerVariant: [String: Double]
    let results: [Result]
}

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let cores = gpuCoreCount()
let options = MTLCompileOptions()
options.mathMode = .safe
options.languageVersion = .version3_1

// Operand and accumulator values are multiples of 1/64 in [-1, 1]: exact in every type,
// so all variants multiply identical data.
var state: UInt64 = 0x2545F4914F6CDD1D
func nextValue() -> Float {
    state = state &* 6364136223846793005 &+ 1442695040888963407
    return Float(Int((state >> 33) % 129) - 64) / 64
}
let aValues = (0..<(fragments * 64)).map { _ in nextValue() }
let bValues = (0..<(fragments * 64)).map { _ in nextValue() }

func operandBuffer(_ values: [Float], _ type: Scalar) -> MTLBuffer {
    let bytes = values.flatMap { type.encode($0) }
    return device.makeBuffer(bytes: bytes, length: bytes.count, options: .storageModeShared)!
}
func initialAccumulators(_ count: Int) -> [Float] { (0..<count).map { _ in nextValue() } }

// Host reference: one MMA per element is a sequential FMA chain over k = 0..7 in f32
// (P1 found this bit-exact for f32 accumulation); a half accumulator is rounded to half
// after every FMA step of that chain.
func reference(_ variant: Variant, _ shape: Shape, simdgroups: Int, iterations: Int, initial: [Float]) -> [Float] {
    let accumulators = shape.na * shape.nb
    var result = initial
    for sg in 0..<simdgroups {
        let local = sg % (threadsPerThreadgroup / 32)
        for x in 0..<shape.na { for y in 0..<shape.nb {
            let base = (sg * accumulators + x * shape.nb + y) * 64
            for it in 0..<iterations {
                let fa = ((local + it * shape.na + x) & (fragments - 1)) * 64
                let fb = ((local + it * shape.nb + y + 3) & (fragments - 1)) * 64
                for r in 0..<8 { for c in 0..<8 {
                    var value = result[base + r * 8 + c]
                    for k in 0..<8 {
                        value = fmaf(aValues[fa + r * 8 + k], bValues[fb + k * 8 + c], value)
                        if variant.accumulator == .half { value = Float(Float16(value)) }
                    }
                    result[base + r * 8 + c] = value
                } }
            }
        } }
    }
    return result
}

func dispatch(_ pipeline: MTLComputePipelineState, _ a: MTLBuffer, _ b: MTLBuffer, _ out: MTLBuffer, threadgroups: Int, iterations: Int) -> Double {
    struct Params { var iterations: UInt32; var mask: UInt32 }
    var params = Params(iterations: UInt32(iterations), mask: UInt32(fragments - 1))
    let command = queue.makeCommandBuffer()!
    let encoder = command.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(a, offset: 0, index: 0)
    encoder.setBuffer(b, offset: 0, index: 1)
    encoder.setBuffer(out, offset: 0, index: 2)
    encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 3)
    encoder.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: threadsPerThreadgroup, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    if let error = command.error { fatalError("command buffer failed: \(error)") }
    return command.gpuEndTime - command.gpuStartTime
}

let variants = [Variant(operand: .half, accumulator: .float), Variant(operand: .bfloat, accumulator: .float),
                Variant(operand: .float, accumulator: .float), Variant(operand: .half, accumulator: .half)]
let shapes = [Shape(na: 1, nb: 1), Shape(na: 1, nb: 2), Shape(na: 2, nb: 2), Shape(na: 2, nb: 4)]
let simdgroupsPerThreadgroup = threadsPerThreadgroup / 32
var results: [Result] = []
for variant in variants {
    let a = operandBuffer(aValues, variant.operand), b = operandBuffer(bValues, variant.operand)
    for shape in shapes {
        let accumulators = shape.na * shape.nb
        let library = try! device.makeLibrary(source: source(variant, shape), options: options)
        let pipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "mma")!)
        precondition(pipeline.maxTotalThreadsPerThreadgroup >= threadsPerThreadgroup)

        // Correctness against the host reference on one threadgroup.
        let checkIterations = 37
        let checkElements = simdgroupsPerThreadgroup * accumulators * 64
        let initial = initialAccumulators(checkElements)
        let checkOut = operandBuffer(initial, variant.accumulator)
        _ = dispatch(pipeline, a, b, checkOut, threadgroups: 1, iterations: checkIterations)
        let expected = reference(variant, shape, simdgroups: simdgroupsPerThreadgroup, iterations: checkIterations, initial: initial)
        var unequal = 0, maxError = 0.0, maxRelative = 0.0
        for i in 0..<checkElements {
            let actual = variant.accumulator.decode(checkOut.contents(), i)
            if actual.bitPattern != expected[i].bitPattern { unequal += 1 }
            let error = abs(Double(actual) - Double(expected[i]))
            maxError = max(maxError, error)
            maxRelative = max(maxRelative, error / max(abs(Double(expected[i])), 1))
        }
        let check = Check(threadgroups: 1, iterations: checkIterations, elements: checkElements, unequalBits: unequal, maxAbsoluteError: maxError,
            maxRelativeError: maxRelative,
            reference: variant.accumulator == .half ? "sequential FMA over k, rounded to half after every FMA step" : "f32 sequential FMA over k per MMA")
        // Rounding-order differences are reported, not rejected; anything larger means
        // the device did not compute the chain it was given.
        precondition(maxRelative < 1e-2, "\(variant.operand)/\(variant.accumulator) \(shape) disagrees with the host reference: \(check)")

        // Throughput over several iteration counts on a saturating grid.
        let threadgroups = cores * threadgroupsPerCore
        let simdgroups = threadgroups * simdgroupsPerThreadgroup
        let out = operandBuffer(initialAccumulators(simdgroups * accumulators * 64), variant.accumulator)
        let flopsPerIteration = Double(simdgroups * accumulators * 2 * 8 * 8 * 8)
        // Iteration counts are interleaved within each repetition so that clock or
        // contention drift affects every point alike instead of bending the slope.
        var times = Array(repeating: [Double](), count: iterationCounts.count)
        for sample in 0..<(warmup + repetitions) {
            for (index, iterations) in iterationCounts.enumerated() {
                let t = dispatch(pipeline, a, b, out, threadgroups: threadgroups, iterations: iterations)
                if sample >= warmup { times[index].append(t) }
            }
        }
        let points = zip(iterationCounts, times).map { iterations, times in
            let seconds = median(times)
            return Point(iterations: iterations, gpuSeconds: times, medianSeconds: seconds, tflops: flopsPerIteration * Double(iterations) / seconds / 1e12)
        }
        let xs = points.map { Double($0.iterations) }, ys = points.map { $0.medianSeconds }
        let mx = xs.reduce(0, +) / Double(xs.count), my = ys.reduce(0, +) / Double(ys.count)
        let slope = zip(xs, ys).map { ($0 - mx) * ($1 - my) }.reduce(0, +) / xs.map { ($0 - mx) * ($0 - mx) }.reduce(0, +)
        let intercept = my - slope * mx
        let residual = zip(xs, ys).map { abs($1 - (intercept + slope * $0)) / $1 }.max()!
        let result = Result(operand: variant.operand, accumulator: variant.accumulator, accumulators: accumulators, shape: shape,
            operandLoadsPerIteration: shape.na + shape.nb, mmasPerIteration: accumulators, threadgroups: threadgroups, simdgroups: simdgroups,
            check: check, points: points, slopeSecondsPerIteration: slope, interceptSeconds: intercept,
            linearityMaxRelativeResidual: residual, tflopsFromSlope: flopsPerIteration / slope / 1e12)
        results.append(result)
        FileHandle.standardError.write(Data(String(format: "%@x%@->%@ acc %d: %.2f TFLOP/s (slope), residual %.3f, check unequal %d max err %.2e\n",
            variant.operand.rawValue, variant.operand.rawValue, variant.accumulator.rawValue, accumulators, result.tflopsFromSlope, residual, unequal, maxError).utf8))
    }
}

var best: [String: Double] = [:]
for result in results {
    let key = "\(result.operand.rawValue)->\(result.accumulator.rawValue)"
    best[key] = max(best[key] ?? 0, result.tflopsFromSlope)
}
let report = Report(probe: "simdgroup_throughput", backend: "metal", host: hostName(), device: device.name, gpuCores: cores,
    operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString, languageVersion: "3.1",
    threadsPerThreadgroup: threadsPerThreadgroup, fragmentsInThreadgroupMemory: fragments, repetitions: repetitions, warmup: warmup,
    definitions: [
        "flops": "2 * 8 * 8 * 8 per simdgroup_multiply_accumulate",
        "tflopsFromSlope": "simdgroups * mmasPerIteration * 1024 / slope of median GPU time vs iteration count; excludes dispatch and staging cost",
        "points.tflops": "whole-dispatch rate including dispatch and threadgroup staging",
        "shape": "per iteration each simdgroup loads na A and nb B fragments from threadgroup memory and issues na*nb MMAs into independent accumulators",
        "linearityMaxRelativeResidual": "max |t - (intercept + slope * iterations)| / t over the iteration counts; small values show the MMAs execute per iteration",
    ],
    bestPerVariant: best, results: results)
let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
FileHandle.standardOutput.write(try! encoder.encode(report))
FileHandle.standardOutput.write(Data("\n".utf8))
