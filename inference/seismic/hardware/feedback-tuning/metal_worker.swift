// Standalone experimental Metal observer. This is not a Seismic backend.
import Foundation
import Metal

enum ProbeError: Error { case invalid(String) }
func now() -> Double { ProcessInfo.processInfo.systemUptime }
func milliseconds(_ start: Double) -> Double { (now() - start) * 1000 }
func emit(_ value: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
    print(String(data: data, encoding: .utf8)!); fflush(stdout)
}
func median(_ values: [Double]) -> Double {
    let a = values.sorted(), m = a.count / 2
    return a.count % 2 == 0 ? (a[m - 1] + a[m]) / 2 : a[m]
}
func coefficient(_ stage: Int) -> (Float, Float) {
    (1.0003 + Float(stage) * 0.00017, stage % 2 == 0 ? 0.003 : -0.004)
}
func literal(_ v: Float) -> String { String(format: "%.9ef", Double(v)) }

struct Command {
    let pipeline: MTLComputePipelineState
    let input: MTLBuffer
    let output: MTLBuffer
    let groups: Int
    let threads: Int
}

final class Observer {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let fixture: String
    let stages: Int
    let n = 1 << 20
    let width = 1024
    let input: MTLBuffer
    let scratch: [MTLBuffer]
    let expected: [Float]
    let options = MTLCompileOptions()
    var artifacts: [String: MTLComputePipelineState] = [:]
    var compileMS = 0.0
    var newArtifacts = 0

    init(fixture: String) throws {
        guard ["chain", "reduce", "stencil"].contains(fixture) else { throw ProbeError.invalid("unknown fixture") }
        guard let d = MTLCreateSystemDefaultDevice(), let q = d.makeCommandQueue() else {
            throw ProbeError.invalid("Metal unavailable")
        }
        device = d; queue = q; self.fixture = fixture; stages = fixture == "reduce" ? 6 : 8
        guard let x = d.makeBuffer(length: n * 4, options: .storageModeShared),
              let a = d.makeBuffer(length: n * 4, options: .storageModeShared),
              let b = d.makeBuffer(length: n * 4, options: .storageModeShared) else {
            throw ProbeError.invalid("buffer allocation")
        }
        input = x; scratch = [a, b]
        options.fastMathEnabled = false
        var ref = [Float](repeating: 0, count: n)
        let p = x.contents().bindMemory(to: Float.self, capacity: n)
        for i in 0..<n {
            let value = Float((i * 17 + 31) % 2048) / 1024 - 1
            p[i] = value; ref[i] = value
        }
        // Host reference keeps each primitive in Float, separately from MSL generation.
        for stage in 0..<stages {
            let (a, b) = coefficient(stage)
            let previous = ref
            for i in 0..<n {
                let value: Float
                if fixture == "stencil" && stage % 2 == 1 {
                    let left = previous[(i + n - 1) % n], right = previous[(i + 1) % n]
                    value = (previous[i] * 0.5).addingProduct(left, 0.25).addingProduct(right, 0.25)
                } else { value = previous[i] }
                let v = b.addingProduct(value, a)
                ref[i] = v / (1 + abs(v) * 0.03125)
            }
        }
        if fixture == "reduce" {
            for row in 0..<(n / width) {
                var sum = 0.0
                for j in 0..<width { let v = Double(ref[row * width + j]); sum += v * v }
                let denominator = Float(sqrt(sum / Double(width) + 0.00001))
                for j in 0..<width { ref[row * width + j] /= denominator }
            }
        }
        expected = ref
    }

    var axes: [Int] {
        Array(repeating: 2, count: stages - 1) + [4, 4] + (fixture == "reduce" ? [4] : [])
    }

    func pipeline(key: String, source: String) throws -> MTLComputePipelineState {
        if let p = artifacts[key] { return p }
        let start = now()
        let library = try device.makeLibrary(source: source, options: options)
        guard let function = library.makeFunction(name: "candidate") else { throw ProbeError.invalid("entry missing") }
        let p = try device.makeComputePipelineState(function: function)
        compileMS += milliseconds(start); newArtifacts += 1; artifacts[key] = p
        return p
    }

    func segment(start: Int, end: Int, lanes: Int) throws -> MTLComputePipelineState {
        var body = ""
        var functions = ""
        for stage in start..<end {
            let (a, b) = coefficient(stage)
            body += "v=fma(v,\(literal(a)),\(literal(b))); v=v/(1.0f+fabs(v)*0.03125f);\n"
            let previous = stage == start ? "x[id]" : "stage\(stage-1)(x,id)"
            let left = stage == start ? "x[(id+\(n-1))&\(n-1)]" : "stage\(stage-1)(x,(id+\(n-1))&\(n-1))"
            let right = stage == start ? "x[(id+1)&\(n-1)]" : "stage\(stage-1)(x,(id+1)&\(n-1))"
            let value = fixture == "stencil" && stage % 2 == 1
                ? "fma(\(right),0.25f,fma(\(left),0.25f,\(previous)*0.5f))" : previous
            functions += "inline float stage\(stage)(device const float* x,uint id){float v=\(value);v=fma(v,\(literal(a)),\(literal(b)));return v/(1.0f+fabs(v)*0.03125f);}\n"
        }
        let laneBody = fixture == "stencil" ? "float v=stage\(end-1)(x,id);" : "float v=x[id];\(body)"
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        \(fixture == "stencil" ? functions : "")
        kernel void candidate(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
        #pragma unroll
          for(uint lane=0;lane<\(lanes);++lane){
            uint id=tid*\(lanes)+lane;
            if(id<\(n)){\(laneBody)y[id]=v;}
          }
        }
        """
        return try pipeline(key: "\(fixture)-segment-\(start)-\(end)-\(lanes)", source: source)
    }

    func reduction(threads: Int) throws -> MTLComputePipelineState {
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void candidate(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], uint lane [[thread_index_in_threadgroup]], uint row [[threadgroup_position_in_grid]]) {
          threadgroup float partial[\(threads)]; float sum=0;
          for(uint j=lane;j<\(width);j+=\(threads)){float v=x[row*\(width)+j];sum=fma(v,v,sum);}
          partial[lane]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
          for(uint step=\(threads/2);step>0;step/=2){if(lane<step)partial[lane]+=partial[lane+step];threadgroup_barrier(mem_flags::mem_threadgroup);}
          float denom=sqrt(partial[0]/\(width).0f+0.00001f);
          for(uint j=lane;j<\(width);j+=\(threads))y[row*\(width)+j]=x[row*\(width)+j]/denom;
        }
        """
        return try pipeline(key: "reduction-\(threads)", source: source)
    }

    func realize(_ point: [Int]) throws -> [Command] {
        guard point.count == axes.count, zip(point, axes).allSatisfy({ $0 >= 0 && $0 < $1 }) else {
            throw ProbeError.invalid("coordinate outside declared domain")
        }
        let lanes = [1, 2, 4, 8][point[stages - 1]]
        let threads = [64, 128, 256, 512][point[stages]]
        var commands: [Command] = [], lower = 0
        for upper in 1...stages {
            // 0 keeps a materialization; 1 fuses across this boundary.
            if upper == stages || point[upper - 1] == 0 {
                let p = try segment(start: lower, end: upper, lanes: lanes)
                guard threads <= p.maxTotalThreadsPerThreadgroup else { throw ProbeError.invalid("illegal launch") }
                commands.append(Command(pipeline: p, input: commands.last?.output ?? input,
                    output: scratch[commands.count % 2], groups: (n + lanes * threads - 1) / (lanes * threads), threads: threads))
                lower = upper
            }
        }
        if fixture == "reduce" {
            let threads = [64, 128, 256, 512][point[stages + 1]]
            let p = try reduction(threads: threads)
            guard threads <= p.maxTotalThreadsPerThreadgroup else { throw ProbeError.invalid("illegal reduction launch") }
            commands.append(Command(pipeline: p, input: commands.last!.output,
                output: scratch[commands.count % 2], groups: n / width, threads: threads))
        }
        return commands
    }

    func execute(_ commands: [Command], repetitions: Int) throws -> Double {
        guard let cb = queue.makeCommandBuffer(), let encoder = cb.makeComputeCommandEncoder() else {
            throw ProbeError.invalid("command allocation")
        }
        for _ in 0..<repetitions {
            for c in commands {
                encoder.setComputePipelineState(c.pipeline)
                encoder.setBuffer(c.input, offset: 0, index: 0)
                encoder.setBuffer(c.output, offset: 0, index: 1)
                encoder.dispatchThreadgroups(MTLSize(width: c.groups, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: c.threads, height: 1, depth: 1))
            }
        }
        encoder.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        if let error = cb.error { throw error }
        let duration = (cb.gpuEndTime - cb.gpuStartTime) * 1000
        guard duration > 0 else { throw ProbeError.invalid("invalid GPU timestamp") }
        return duration / Double(repetitions)
    }

    func observe(_ point: [Int], trials: Int, targetMS: Double) throws -> [String: Any] {
        guard (1...32).contains(trials), targetMS > 0, targetMS <= 100 else { throw ProbeError.invalid("invalid observation protocol") }
        let start = now(), before = compileMS, oldCount = newArtifacts
        let commands = try realize(point)
        _ = try execute(commands, repetitions: 1)
        let pilot = try execute(commands, repetitions: 3)
        let repetitions = min(128, max(1, Int(ceil(targetMS / pilot))))
        _ = try execute(commands, repetitions: repetitions)
        var samples: [Double] = []
        for _ in 0..<trials { samples.append(try execute(commands, repetitions: repetitions)) }
        let values = commands.last!.output.contents().bindMemory(to: Float.self, capacity: n)
        var maxError: Float = 0
        for i in 0..<n {
            let error = abs(values[i] - expected[i])
            guard values[i].isFinite, error <= 0.0002 * max(1, abs(expected[i])) else {
                throw ProbeError.invalid("numerical mismatch at \(i): \(values[i]) != \(expected[i])")
            }
            maxError = max(maxError, error)
        }
        return ["point": point, "samples_ms": samples, "median_ms": median(samples),
            "compile_ms": compileMS - before, "new_artifacts": newArtifacts - oldCount,
            "artifact_count": artifacts.count, "commands": commands.count,
            "repetitions": repetitions, "wall_ms": milliseconds(start), "max_error": maxError,
            "checked_elements": n]
    }
}

do {
    let start = now(), fixture = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "chain"
    let observer = try Observer(fixture: fixture)
    emit(["kind": "ready", "fixture": fixture, "axes": observer.axes, "device": observer.device.name,
          "os": ProcessInfo.processInfo.operatingSystemVersionString, "initialization_ms": milliseconds(start),
          "elements": observer.n, "stages": observer.stages, "endpoint": "complete command-buffer GPU duration per replay"])
    while let line = readLine() {
        do {
            guard let data = line.data(using: .utf8),
                  let request = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let point = request["point"] as? [Int] else { throw ProbeError.invalid("malformed request") }
            let result = try observer.observe(point, trials: request["trials"] as? Int ?? 5,
                                              targetMS: request["target_ms"] as? Double ?? 6)
            emit(result)
        } catch { emit(["error": String(describing: error)]) }
    }
} catch { emit(["error": String(describing: error)]); exit(1) }
