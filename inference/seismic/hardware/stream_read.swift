// P3/R6 streaming-read bandwidth probe (Metal).
// Each lane issues UNROLL independent 16 B (uint4) loads per step of a grid-stride
// loop, adds them into a register accumulator, and each simdgroup writes one
// simd_sum. The sum of all outputs equals the 32-bit wrapping sum of every word in
// the buffer, which is checked for every configuration: each byte is read exactly
// once and no read can be eliminated.
//
//   swiftc -O stream_read.swift -o stream-read
//   ./stream-read [--sizes-gb 1,2,4] [--repetitions 10] > bandwidth-<host>.json
import Foundation
import Metal
import IOKit

struct Arguments {
    var sizesGB: [Int] = [1, 2, 4]
    var repetitions = 10
    var warmup = 2
    var unrolls = [1, 2, 4, 8]
    var threadsPerThreadgroup = [64, 128, 256, 512, 1024]
    var gridMultipliers: [Double] = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]
}
func parseList(_ text: String) -> [Int] { text.split(separator: ",").map { Int($0)! } }
func parseArguments() -> Arguments {
    var result = Arguments()
    var iterator = CommandLine.arguments.dropFirst().makeIterator()
    while let flag = iterator.next() {
        guard let value = iterator.next() else { fatalError("\(flag) requires a value") }
        switch flag {
        case "--sizes-gb": result.sizesGB = parseList(value)
        case "--repetitions": result.repetitions = Int(value)!
        case "--warmup": result.warmup = Int(value)!
        case "--unrolls": result.unrolls = parseList(value)
        case "--threads-per-threadgroup": result.threadsPerThreadgroup = parseList(value)
        case "--grid-multipliers": result.gridMultipliers = value.split(separator: ",").map { Double($0)! }
        default: fatalError("unknown flag \(flag)")
        }
    }
    precondition(result.repetitions >= 10, "at least 10 timed repetitions are required")
    return result
}

// Apple does not expose the GPU core count through Metal; the AGX accelerator's
// IORegistry entry publishes it as `gpu-core-count`.
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

struct Configuration: Codable {
    let bufferBytes: Int
    let unroll: Int
    let threadsPerThreadgroup: Int
    let grid: String
    let threadgroups: Int
    let gridThreads: Int
    let bytesInFlightIssued: Int
    let gpuSeconds: [Double]
    let medianSeconds: Double
    let gbPerSecond: Double
    let checksumCorrect: Bool
}
struct Best: Codable {
    let bufferBytes: Int
    let configuration: Configuration
}
struct Report: Codable {
    let probe: String
    let backend: String
    let host: String
    let device: String
    let gpuCores: Int
    let operatingSystem: String
    let maxBufferLength: Int
    let storageMode: String
    let parameters: Parameters
    let definitions: [String: String]
    let bestPerSize: [Best]
    let best: Best
    let results: [Configuration]
}
struct Parameters: Codable {
    let sizesGB: [Int]
    let unrolls: [Int]
    let threadsPerThreadgroup: [Int]
    let gridMultipliers: [Double]
    let repetitions: Int
    let warmup: Int
}

let arguments = parseArguments()
let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let cores = gpuCoreCount()
let options = MTLCompileOptions()
options.mathMode = .safe

let fillSource = """
#include <metal_stdlib>
using namespace metal;
kernel void fill(device uint* data [[buffer(0)]], uint id [[thread_position_in_grid]]) {
  data[id] = id * 0x9E3779B1u;
}
"""
func readSource(unroll: Int) -> String {
    let loads = (0..<unroll).map { "v[\($0)] = src[i + \($0)u * grid];" }.joined(separator: "\n      ")
    let sums = (0..<unroll).map { "acc += v[\($0)];" }.joined(separator: "\n      ")
    return """
    #include <metal_stdlib>
    using namespace metal;
    kernel void stream_read(device const uint4* src [[buffer(0)]], device uint* out [[buffer(1)]],
                            constant uint& count [[buffer(2)]],
                            uint id [[thread_position_in_grid]], uint grid [[threads_per_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
      uint4 acc = uint4(0);
      uint i = id;
      for (; i + \(unroll - 1)u * grid < count; i += \(unroll)u * grid) {
        uint4 v[\(unroll)];
        \(loads)
        \(sums)
      }
      for (; i < count; i += grid) { acc += src[i]; }
      uint total = simd_sum(acc.x + acc.y + acc.z + acc.w);
      if (lane == 0) { out[id / 32] = total; }
    }
    """
}
func pipeline(_ source: String, _ name: String) -> MTLComputePipelineState {
    let library = try! device.makeLibrary(source: source, options: options)
    return try! device.makeComputePipelineState(function: library.makeFunction(name: name)!)
}
let fillPipeline = pipeline(fillSource, "fill")
let readPipelines = Dictionary(uniqueKeysWithValues: arguments.unrolls.map { ($0, pipeline(readSource(unroll: $0), "stream_read")) })

func run(_ encode: (MTLComputeCommandEncoder) -> Void) -> Double {
    let command = queue.makeCommandBuffer()!
    let encoder = command.makeComputeCommandEncoder()!
    encode(encoder)
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    if let error = command.error { fatalError("command buffer failed: \(error)") }
    return command.gpuEndTime - command.gpuStartTime
}
func median(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    let middle = sorted.count / 2
    return sorted.count % 2 == 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2
}

var results: [Configuration] = []
var bestPerSize: [Best] = []
for sizeGB in arguments.sizesGB {
    let bytes = sizeGB << 30
    precondition(bytes <= device.maxBufferLength, "\(sizeGB) GB exceeds maxBufferLength \(device.maxBufferLength)")
    let words = bytes / 4
    var vectors = UInt32(bytes / 16)
    precondition(UInt64(vectors) * 9 < UInt64(UInt32.max), "32-bit loop indices require a smaller buffer")
    // GPU-resident storage, filled (and therefore paged in) by the GPU before timing.
    let source = device.makeBuffer(length: bytes, options: .storageModePrivate)!
    _ = run { encoder in
        encoder.setComputePipelineState(fillPipeline)
        encoder.setBuffer(source, offset: 0, index: 0)
        encoder.dispatchThreads(MTLSize(width: words, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    // Sum over w of w * 0x9E3779B1 (mod 2^32) = 0x9E3779B1 * n(n-1)/2 (mod 2^32).
    let n = UInt64(words)
    let expected = UInt32(truncatingIfNeeded: (n * (n - 1) / 2) &* 0x9E3779B1)
    var sizeResults: [Configuration] = []
    for unroll in arguments.unrolls {
        let read = readPipelines[unroll]!
        for tpg in arguments.threadsPerThreadgroup where tpg <= read.maxTotalThreadsPerThreadgroup {
            let full = (Int(vectors) + unroll * tpg - 1) / (unroll * tpg)
            var grids = arguments.gridMultipliers.map { ("\($0)xcores", max(1, Int($0 * Double(cores)))) }.filter { $0.1 < full }
            grids.append(("full", full))
            for (label, threadgroups) in grids {
                let gridThreads = threadgroups * tpg
                let output = device.makeBuffer(length: gridThreads / 32 * 4, options: .storageModeShared)!
                var times: [Double] = []
                for sample in 0..<(arguments.warmup + arguments.repetitions) {
                    let t = run { encoder in
                        encoder.setComputePipelineState(read)
                        encoder.setBuffer(source, offset: 0, index: 0)
                        encoder.setBuffer(output, offset: 0, index: 1)
                        encoder.setBytes(&vectors, length: 4, index: 2)
                        encoder.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
                    }
                    if sample >= arguments.warmup { times.append(t) }
                }
                let sums = output.contents().bindMemory(to: UInt32.self, capacity: gridThreads / 32)
                var total: UInt32 = 0
                for s in 0..<(gridThreads / 32) { total &+= sums[s] }
                let seconds = median(times)
                let configuration = Configuration(bufferBytes: bytes, unroll: unroll, threadsPerThreadgroup: tpg, grid: label,
                    threadgroups: threadgroups, gridThreads: gridThreads, bytesInFlightIssued: gridThreads * unroll * 16,
                    gpuSeconds: times, medianSeconds: seconds, gbPerSecond: Double(bytes) / seconds / 1e9, checksumCorrect: total == expected)
                precondition(configuration.checksumCorrect, "checksum mismatch: \(configuration)")
                sizeResults.append(configuration)
                FileHandle.standardError.write(Data(String(format: "%d GB unroll %d tpg %d %@: %.1f GB/s\n", sizeGB, unroll, tpg, label, configuration.gbPerSecond).utf8))
            }
        }
    }
    bestPerSize.append(Best(bufferBytes: bytes, configuration: sizeResults.max { $0.gbPerSecond < $1.gbPerSecond }!))
    results += sizeResults
}

let report = Report(probe: "stream_read", backend: "metal", host: hostName(), device: device.name, gpuCores: cores,
    operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString, maxBufferLength: device.maxBufferLength, storageMode: "private",
    parameters: Parameters(sizesGB: arguments.sizesGB, unrolls: arguments.unrolls, threadsPerThreadgroup: arguments.threadsPerThreadgroup,
        gridMultipliers: arguments.gridMultipliers, repetitions: arguments.repetitions, warmup: arguments.warmup),
    definitions: [
        "gbPerSecond": "bufferBytes / median(gpuEndTime - gpuStartTime) / 1e9, one dispatch per command buffer",
        "bytesInFlightIssued": "gridThreads * unroll * 16: independent 16 B loads each lane can have outstanding at once, summed over the whole grid. Metal exposes no occupancy query, so this is not capped by resident threads; it is exact only when the grid fits on the GPU at once",
        "grid": "Nxcores = max(1, floor(N * gpuCores)) threadgroups (grid-stride loop); full = one step per lane covering the buffer",
        "checksumCorrect": "wrapping sum of all simdgroup outputs equals the wrapping sum of every 32-bit word",
    ],
    bestPerSize: bestPerSize, best: bestPerSize.max { $0.configuration.gbPerSecond < $1.configuration.gbPerSecond }!, results: results)
let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
FileHandle.standardOutput.write(try! encoder.encode(report))
FileHandle.standardOutput.write(Data("\n".utf8))
