// Independent native-mapping inspection. This tool neither ranks compiler
// candidates nor supplies timing/resource parameters to Seismic selection.
import Foundation
import Metal
import CryptoKit

guard CommandLine.arguments.count == 4 else {
    fatalError("usage: metal-archive SOURCE.METAL FUNCTION OUTPUT_DIRECTORY")
}
let sourceURL = URL(fileURLWithPath: CommandLine.arguments[1])
let name = CommandLine.arguments[2]
let destination = URL(fileURLWithPath: CommandLine.arguments[3], isDirectory: true)
try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
guard let device = MTLCreateSystemDefaultDevice() else { fatalError("Metal device unavailable") }
let sourceBytes = try Data(contentsOf: sourceURL)
guard let source = String(data: sourceBytes, encoding: .utf8) else { fatalError("source is not UTF-8") }
let options = MTLCompileOptions()
options.fastMathEnabled = false // Match the production runtime's numerical setting.
let library = try device.makeLibrary(source: source, options: options)
guard let function = library.makeFunction(name: name) else { fatalError("function unavailable: \(name)") }
let archive = try device.makeBinaryArchive(descriptor: MTLBinaryArchiveDescriptor())
let descriptor = MTLComputePipelineDescriptor()
descriptor.computeFunction = function
descriptor.binaryArchives = [archive]
try archive.addComputePipelineFunctions(descriptor: descriptor)
let pipeline = try device.makeComputePipelineState(descriptor: descriptor, options: [], reflection: nil)
let archiveURL = destination.appendingPathComponent("native.metalar")
try archive.serialize(to: archiveURL)
let archiveBytes = try Data(contentsOf: archiveURL)
func digest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
let report: [String: Any] = [
    "device": device.name,
    "registry_id": String(device.registryID),
    "operating_system": ProcessInfo.processInfo.operatingSystemVersionString,
    "function": name,
    "source_sha256": digest(sourceBytes),
    "archive_sha256": digest(archiveBytes),
    "archive_bytes": archiveBytes.count,
    "fast_math_enabled": false,
    "thread_execution_width": pipeline.threadExecutionWidth,
    "max_total_threads_per_threadgroup": pipeline.maxTotalThreadsPerThreadgroup,
    "static_threadgroup_memory_bytes": pipeline.staticThreadgroupMemoryLength,
    "qualified_primitive_mapping": false,
    "limitations": ["Binary capture only; instruction decoding and resource interpretation require independent validation.",
                    "Pipeline thread limits are not resident-threadgroup capacity or register/spill counts."]
]
try sourceBytes.write(to: destination.appendingPathComponent("source.metal"))
let encoded = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
try encoded.write(to: destination.appendingPathComponent("report.json"))
FileHandle.standardOutput.write(encoded)
FileHandle.standardOutput.write(Data("\n".utf8))
