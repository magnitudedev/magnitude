// Fixed-probe characterization only. Never inspect a qualification candidate.
import Foundation
import Metal
let root=URL(fileURLWithPath:CommandLine.arguments[1])
let device=MTLCreateSystemDefaultDevice()!,options=MTLCompileOptions()
options.fastMathEnabled=false
let library=try device.makeLibrary(source:String(contentsOf:root.appendingPathComponent("probes.metal"),encoding:.utf8),options:options)
for name in CommandLine.arguments.dropFirst(2) {
    var reflection:MTLComputePipelineReflection?
    let pipeline=try device.makeComputePipelineState(function:library.makeFunction(name:name)!,options:.argumentInfo,reflection:&reflection)
    let arguments=reflection!.arguments.map{["name":$0.name,"index":$0.index,"type":$0.type.rawValue,"active":$0.isActive] as [String:Any]}
    let data=try JSONSerialization.data(withJSONObject:["kernel":name,"arguments":arguments,"static_shared":pipeline.staticThreadgroupMemoryLength],options:.sortedKeys)
    print(String(data:data,encoding:.utf8)!)
}
