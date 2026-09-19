import { Effect, Schema } from "effect"
import { expect, test } from "vitest"
import { LoadedBackendModule, NativeExecution } from "../src/execution-telemetry"
import { attestRuntimeModules } from "../src/runtime-modules"

const cpu = Schema.decodeUnknownSync(LoadedBackendModule)({ name: "libggml-cpu.so", sha256: "a".repeat(64), bytes: 64 })
const metal = Schema.decodeUnknownSync(LoadedBackendModule)({ name: "libggml-metal.so", sha256: "b".repeat(64), bytes: 128 })
const execution = (modules?: readonly (typeof LoadedBackendModule.Type)[]) => Schema.decodeUnknownSync(NativeExecution)({
  traceId: "a".repeat(32), model: "fixture", workerPid: 42, workerGeneration: "1", requestId: "2",
  allocations: [{ kind: "host", model_bytes: 1024 }], ...(modules === undefined ? {} : { modules }),
})

test("loaded modules must uniquely match the admitted archive bytes, not just filenames", () => Effect.runPromise(Effect.gen(function* () {
  yield* attestRuntimeModules(execution([cpu, metal]), [cpu, metal])
  // Installed CPU variants need not all be loaded by the current hardware.
  yield* attestRuntimeModules(execution([cpu]), [cpu, metal])
  for (const observed of [execution(), execution([]), execution([cpu, cpu]), execution([{ ...cpu, bytes: 65 }]),
    execution([{ ...cpu, sha256: metal.sha256 }]), execution([{ ...cpu, name: "unexpected.so" }])]) {
    expect((yield* attestRuntimeModules(observed, [cpu, metal]).pipe(Effect.either))._tag).toBe("Left")
  }
  expect((yield* attestRuntimeModules(execution([cpu]), [cpu, cpu]).pipe(Effect.either))._tag).toBe("Left")
})))
