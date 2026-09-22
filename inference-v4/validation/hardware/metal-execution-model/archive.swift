// Observation-side only: inspect compiled fixed probes, never supply candidate facts.
import Foundation
import Metal
let root=URL(fileURLWithPath:CommandLine.arguments[1])
let device=MTLCreateSystemDefaultDevice()!,opts=MTLCompileOptions();opts.fastMathEnabled=false
let library=try device.makeLibrary(source:String(contentsOf:root.appendingPathComponent("probes.metal"),encoding:.utf8),options:opts)
let out=root.appendingPathComponent("archives")
try FileManager.default.createDirectory(at:out,withIntermediateDirectories:true)
let selected=Set(CommandLine.arguments.dropFirst(2))
for name in library.functionNames.sorted() where selected.isEmpty || selected.contains(name) {
 do {
  let archive=try device.makeBinaryArchive(descriptor:MTLBinaryArchiveDescriptor())
  let descriptor=MTLComputePipelineDescriptor();descriptor.computeFunction=library.makeFunction(name:name)!
  try archive.addComputePipelineFunctions(descriptor:descriptor)
  try archive.serialize(to:out.appendingPathComponent(name+".bin"))
  print(name)
 } catch {print("ERROR \(name): \(error)")}
}
