import Foundation
import Metal
import Darwin

let root=URL(fileURLWithPath:CommandLine.arguments[1])
let phase=CommandLine.arguments[2]
let label=CommandLine.arguments[3]
let warmed=CommandLine.arguments.contains("--warm")
func now()->Double { ProcessInfo.processInfo.systemUptime }
func log(_ row:[String:Any]) { let d=try! JSONSerialization.data(withJSONObject:row,options:[.sortedKeys]); print(String(data:d,encoding:.utf8)!); fflush(stdout) }
let begin=now()
let device=MTLCreateSystemDefaultDevice()!, queue=device.makeCommandQueue()!
let manifest=try JSONSerialization.jsonObject(with:Data(contentsOf:root.appendingPathComponent("manifest.json"))) as! [[String:Any]]
let rows=manifest.filter { phase=="all" || ($0["split"] as! String)==phase || (phase=="diagnostic" && ($0["split"] as! String)=="diagnostic") }
log(["kind":"environment","label":label,"phase":phase,"device":device.name,"registry_id":String(device.registryID),"os":ProcessInfo.processInfo.operatingSystemVersionString,"max_threadgroup_memory":device.maxThreadgroupMemoryLength,"recommended_working_set":device.recommendedMaxWorkingSetSize,"counter_sets":device.counterSets?.map{$0.name} ?? [],"endpoint":"command-buffer GPU start/end; host submit-to-completion","thermal_state":ProcessInfo.processInfo.thermalState.rawValue])
let compile=now(), opts=MTLCompileOptions(); opts.fastMathEnabled=false
let source=try String(contentsOf:root.appendingPathComponent("probes.metal"),encoding:.utf8)
let library=try device.makeLibrary(source:source,options:opts)
log(["kind":"library","seconds":now()-compile,"source_bytes":source.utf8.count])
var pipelines:[String:MTLComputePipelineState]=[:]
func pipelineKey(_ c:[String:Any])->String {let name=c["kernel"] as! String;return (c["compile_max_threads"] as? Int).map{name+"__cap\($0)"} ?? name}
var compilation:[String:(String,Int)]=["fma_c16":("fma_c16",0)]
for c in manifest {compilation[pipelineKey(c)]=(c["kernel"] as! String,c["compile_max_threads"] as? Int ?? 0)}
for key in compilation.keys.sorted() {
    let (name,cap)=compilation[key]!
    let start=now()
    do { let p:MTLComputePipelineState
        if cap>0 {let d=MTLComputePipelineDescriptor();d.computeFunction=library.makeFunction(name:name)!;d.maxTotalThreadsPerThreadgroup=cap;p=try device.makeComputePipelineState(descriptor:d,options:[],reflection:nil)}
        else {p=try device.makeComputePipelineState(function:library.makeFunction(name:name)!)}
        pipelines[key]=p
        log(["kind":"pipeline","name":name,"key":key,"compile_max_threads":cap,"seconds":now()-start,"max_threads":p.maxTotalThreadsPerThreadgroup,"execution_width":p.threadExecutionWidth,"static_shared":p.staticThreadgroupMemoryLength])
    } catch { log(["kind":"failure","stage":"pipeline","name":name,"error":String(describing:error)]) }
}
let allocation=now(), inputCount=67108864, outputCount=16777216
let input=device.makeBuffer(length:inputCount*4,options:.storageModeShared)!, output=device.makeBuffer(length:outputCount*4,options:.storageModeShared)!
let xp=input.contents().bindMemory(to:Float.self,capacity:inputCount), yp=output.contents().bindMemory(to:Float.self,capacity:outputCount)
let xu=input.contents().bindMemory(to:UInt32.self,capacity:inputCount), yu=output.contents().bindMemory(to:UInt32.self,capacity:outputCount)
for i in 0..<inputCount { xp[i]=Float(i%251+1)/256 }
memset(output.contents(),0,outputCount*4)
log(["kind":"buffers","seconds":now()-allocation,"input_bytes":input.length,"output_bytes":output.length])
struct Params { var n:UInt32; var it:UInt32; var stride:UInt32; var mode:UInt32; var a:Float; var b:Float; var groups:UInt32; var pad:UInt32=0 }
let warmInput=device.makeBuffer(length:65536*16*4,options:.storageModeShared)!,warmOutput=device.makeBuffer(length:65536*4,options:.storageModeShared)!
let wp=warmInput.contents().bindMemory(to:Float.self,capacity:65536*16)
for i in 0..<(65536*16){wp[i]=0.5}
func preheat()throws {
    if !warmed{return}
    var params=Params(n:65536,it:256,stride:1,mode:0,a:0.99991,b:0.00001,groups:65536)
    let command=queue.makeCommandBuffer()!,enc=command.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipelines["fma_c16"]!);enc.setBuffer(warmInput,offset:0,index:0);enc.setBuffer(warmOutput,offset:0,index:1)
    enc.setBytes(&params,length:MemoryLayout<Params>.stride,index:2);enc.setThreadgroupMemoryLength(16,index:0)
    for _ in 0..<32{enc.dispatchThreads(MTLSize(width:65536,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:128,height:1,depth:1))}
    enc.endEncoding();command.commit();command.waitUntilCompleted();if let e=command.error{throw e}
}
log(["kind":"protocol","warmed":warmed,"warmup":"32 fixed c16 FMA dispatches before every observation; separate buffers and excluded GPU endpoint","rounds":7])
func num(_ c:[String:Any],_ k:String)->Int { (c[k] as! NSNumber).intValue }
func prepare(_ c:[String:Any]) {
    let family=c["family"] as! String,n=num(c,"n"),threads=num(c,"threads")
    if family=="chase" { for i in 0..<n { xu[i]=UInt32((i*5+1)&(n-1)) } }
    else if family=="strict" {
        let mode=num(c,"mode")
        for i in 0..<n {
            let normal:UInt32=0x3f000000|UInt32((i*7919)&0x7fffff)
            if mode>=7 {
                let kind=mode==7 ? i%4 : (i/32)%4
                xu[i]=[UInt32(0x3fa00000),0x123,0,0x7f800000][kind]
                xu[i+n]=[UInt32(0x3fe00000),0x456,0,0x3f800000][kind]
                continue
            }
            let kind=mode<5 ? mode : (mode==5 ? i%5 : (i/32)%5)
            switch kind {
            case 0: xu[i]=normal; xu[i+n]=normal+0x00100000
            case 1: xu[i]=normal; xu[i+n]=normal|0x80000000
            case 2: xu[i]=UInt32(i%1023+1); xu[i+n]=UInt32((i*3)%1023+1)
            case 3: xu[i]=0; xu[i+n]=0
            default: xu[i]=i%2==0 ? 0x7fc00001 : 0x7f800000; xu[i+n]=0x3f800000
            }
        }
    } else if family=="matrix" { for i in 0..<n { xp[i]=0.125 } }
    else {
        let required=family=="stream" ? n*num(c,"stride") : max(n*(c["chains"] as? Int ?? 4),threads)
        for i in 0..<min(inputCount,required) { xp[i]=Float(i%251+1)/256 }
    }
}
func trial(_ c:[String:Any],_ reps:Int)throws->(Double,Double) {
    try preheat()
    let p=pipelines[pipelineKey(c)]!, threads=num(c,"threads"),tg=num(c,"tg")
    var params=Params(n:UInt32(num(c,"n")),it:UInt32(num(c,"it")),stride:UInt32(num(c,"stride")),mode:UInt32(num(c,"mode")),a:(c["a"] as! NSNumber).floatValue,b:(c["b"] as! NSNumber).floatValue,groups:UInt32(threads))
    let command=queue.makeCommandBuffer()!, enc=command.makeComputeCommandEncoder()!
    enc.setComputePipelineState(p); enc.setBuffer(input,offset:0,index:0); enc.setBuffer(output,offset:0,index:1)
    enc.setBytes(&params,length:MemoryLayout<Params>.stride,index:2);enc.setThreadgroupMemoryLength(num(c,"shared"),index:0)
    let commands=c["commands"] as? Int ?? 1
    for _ in 0..<(reps*commands) { enc.dispatchThreads(MTLSize(width:threads,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:tg,height:1,depth:1)) }
    enc.endEncoding();let start=now();command.commit();command.waitUntilCompleted()
    if let error=command.error { throw error }
    let gpu=(command.gpuEndTime-command.gpuStartTime)/Double(reps)
    guard gpu>0 else { throw NSError(domain:"timer",code:1) }
    return (gpu,(now()-start)/Double(reps))
}
func check(_ c:[String:Any])->[String:Any] {
    let family=c["family"] as! String,n=num(c,"n"),it=num(c,"it"),threads=num(c,"threads")
    var checked=0, failures=0, maxError:Double=0
    func compare(_ actual:Float,_ expected:Float) { checked+=1; let error=Double(abs(actual-expected)); maxError=max(maxError,error.isFinite ? error : 0); if !(actual.isNaN && expected.isNaN) && actual != expected && error > max(0.00002,Double(abs(expected))*0.00002) { failures+=1 } }
    if family=="integer" {
        let chains=num(c,"chains"), op=c["operation"] as! String,mode=UInt32(num(c,"mode"))
        for id in [0,threads/2,threads-1] {
            if op=="sfu" {
                var sum:Float=0
                for k in 0..<chains {var v=xp[id+k*n];for _ in 0..<(it*8){v=1/sqrtf(v+1)};sum+=v};compare(yp[id],sum)
            } else if op=="wide" {
                var sum:UInt64=0
                for k in 0..<chains {var v=UInt64(id+k+1);for _ in 0..<(it*8){v=(v &* UInt64(mode))^(v>>33)};sum=sum &+ v};checked+=1;if yu[id] != UInt32(truncatingIfNeeded:sum){failures+=1}
            } else {
                var sum:UInt32=0
                for k in 0..<chains {var v=UInt32(id+k+1);for _ in 0..<(it*8){v=op=="mul" ? v &* mode : (op=="cmul" ? (v &* 1664525)^(v>>13) : (v &+ mode)^(v>>13))};sum=sum &+ v};checked+=1;if yu[id] != sum{failures+=1}
            }
        }
    } else if family=="compute" || family=="pressure" {
        let chains=num(c,"chains"),a=(c["a"] as! NSNumber).floatValue,b=(c["b"] as! NSNumber).floatValue
        for id in [0,threads/2,threads-1] { var sum:Float=0;for k in 0..<chains { var v=xp[id+k*n];for _ in 0..<(it*(c["unroll"] as? Int ?? 8)){v=fmaf(v,a,b)};sum+=v }; if family=="pressure" {let tg=num(c,"tg");sum+=xp[id/tg*tg+(id%tg+1)%tg]};compare(yp[id],sum) }
    } else if family=="stream" {
        for id in [0,n/2,n-1] { var v=xp[id*num(c,"stride")]; for _ in 0..<num(c,"intensity") {v=fmaf(v,(c["a"] as! NSNumber).floatValue,(c["b"] as! NSNumber).floatValue)};compare(yp[id],v) }
    } else if family=="strict" {
        func bf(_ v:Float)->Float {let b=v.bitPattern;if (b&0x7fffffff)>0x7f800000{return Float.nan};return Float(bitPattern:((b &+ 0x7fff &+ ((b>>16)&1))>>16)<<16)}
        for id in [0,threads/2,threads-1] {
            let k=(id+(it-1)*threads)&(n-1),a=xp[k],b=xp[k+n];var expected:Float=0
            switch c["op"] as! String {
            case "add":expected=a+b
            case "mul":expected=a*b
            case "div":expected=a/b
            case "rem":expected=remainderf(a,b)
            case "fma":expected=fmaf(a,b,a)
            case "cmp":expected=a<b ? 1:0
            case "f16":expected=Float(Float16(a)+Float16(b))
            case "bf16":expected=bf(bf(a)+bf(b))
            default:expected=Float(Float16(a))
            }
            let actual=yp[id+threads];checked+=1
            if !(actual.isNaN && expected.isNaN) && actual.bitPattern != expected.bitPattern {failures+=1}
        }
    } else if family=="mixed" {
        for id in [0,threads/2,threads-1] {
            var v=(0..<4).map{xp[id+$0*n]},q=(0..<4).map{UInt32(id+$0+1)}
            for _ in 0..<it {for k in 0..<max(num(c,"f"),num(c,"u")) {
                if k<num(c,"f"){v[k%4]=fmaf(v[k%4],(c["a"] as! NSNumber).floatValue,(c["b"] as! NSNumber).floatValue)}
                if k<num(c,"u"){q[k%4]=(q[k%4] &* 1664525 &+ 1013904223)^(q[k%4]>>13)}
            }}
            compare(yp[id],v.reduce(0,+));checked+=1;if yu[id+n] != q.reduce(0,^) {failures+=1}
        }
    } else if family=="lifecycle" {
        for id in [0,threads-1]{compare(yp[id],xp[id])}
    } else if family=="chase" {
        for id in [0,threads/2,threads-1] {var q=id&(n-1);for _ in 0..<it{q=(q*5+1)&(n-1)};checked+=1;if yu[id] != UInt32(q){failures+=1} }
    } else if family=="matrix" { for id in [0,n/4-1,n/2-1] {compare(yp[id],Float(it)*0.125)} }
    else if family=="atomic" {
        let expected:Float=c["operation"] != nil ? 0 : Float(threads/n*it)
        for id in Set([0,n/2,n-1]) { compare(c["op"] as! String=="cas" ? yp[id] : Float(yu[id]),expected) }
    } else if family=="sync" {
        for id in [0,threads/2,threads-1] {
            let name=c["op"] as! String
            if name=="barrier" {let tg=num(c,"tg"),start=id/tg*tg,actual=min(tg,threads-start);compare(yp[id],xp[start+(id-start+it)%actual])}
            else if name=="shuffle" {compare(yp[id],xp[id/32*32+(id%32+it)%32])}
            else {var sum:Float=0;for k in 0..<32{sum+=xp[id/32*32+k]};compare(yp[id],sum/32)}
        }
    }
    return ["checked":checked,"failures":failures,"max_absolute_error":maxError,"output_bits":[yu[0],yu[min(threads-1,outputCount-1)]]]
}
// Fixed randomization, independent of observed values.
var seed:UInt64=0x6d6574616c
func shuffled<T>(_ a:[T])->[T] { var b=a;if b.count>1 {for i in stride(from:b.count-1,through:1,by:-1){seed=seed &* 6364136223846793005 &+ 1;b.swapAt(i,Int(seed%UInt64(i+1)))}};return b }
var repetitions:[String:Int]=[:]
let acquisition=now()
for c in rows {
    let id=c["id"] as! String
    guard let p=pipelines[pipelineKey(c)],num(c,"tg")<=p.maxTotalThreadsPerThreadgroup,num(c,"shared")<=device.maxThreadgroupMemoryLength else {log(["kind":"failure","stage":"admission","id":id]);continue}
    prepare(c)
    do {
        memset(output.contents(),0,outputCount*4)
        _=try trial(c,1);memset(output.contents(),0,outputCount*4)
        let pilot=try trial(c,1)
        let reps=(c["family"] as! String)=="atomic" ? 1 : min(64,max(1,Int(0.001/max(pilot.0,1e-9))))
        repetitions[id]=reps
        var row:[String:Any]=["kind":"pilot","id":id,"gpu_s":pilot.0,"host_s":pilot.1,"reps":reps,"check":check(c)]
        if c["counted"] as? Bool == true {let n=num(c,"n"),threads=num(c,"threads");let counts=(0..<threads).map{UInt64(yu[n+$0])};row["attempts"]=counts.reduce(0,+);row["max_lane_attempts"]=counts.max()!}
        log(row)
    }catch{log(["kind":"failure","stage":"pilot","id":id,"error":String(describing:error)])}
}
for round in 0..<7 {
    for c in shuffled(rows) {
        let id=c["id"] as! String
        guard let reps=repetitions[id] else {continue}
        prepare(c)
        if c["family"] as! String=="atomic" {memset(output.contents(),0,num(c,"n")*4)}
        do {let value=try trial(c,reps);log(["kind":"sample","id":id,"round":round,"reps":reps,"gpu_s":value.0,"host_s":value.1,"thermal_state":ProcessInfo.processInfo.thermalState.rawValue])}
        catch {log(["kind":"failure","stage":"sample","id":id,"round":round,"error":String(describing:error)])}
    }
    log(["kind":"round","round":round,"elapsed_s":now()-acquisition])
}
log(["kind":"suite","label":label,"phase":phase,"cases":rows.count,"acquisition_s":now()-acquisition,"total_s":now()-begin])
