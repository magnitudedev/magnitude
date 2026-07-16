import * as BunContext from "@effect/platform-bun/BunContext"
import * as BunRuntime from "@effect/platform-bun/BunRuntime"
import { Console, Data, Effect, Schema } from "effect"
import { LlamaBinary, makeLlamaCli, validateLlamaBinary } from "../src/llamacpp"

class MissingExecutable extends Data.TaggedError("MissingExecutable")<{}> {}
const JsonOutput = Schema.parseJson(Schema.Unknown, { space: 2 })

const program = Effect.gen(function* () {
  const executable = process.argv[2]
  if (executable === undefined) return yield* new MissingExecutable()
  const binary = yield* validateLlamaBinary({ executable, source: "configured" })
  const cli = yield* makeLlamaCli().pipe(
    Effect.provideService(LlamaBinary, binary),
  )
  const capabilities = yield* cli.capabilities
  const devices = capabilities.listDevices ? yield* cli.listDevices.pipe(Effect.either) : undefined
  const report = {
    verifiedAt: new Date().toISOString(), executable: binary.executable, fingerprint: binary.fingerprint, build: binary.build,
    capabilities: { fitPrint: capabilities.fitPrint, listDevices: capabilities.listDevices, apiKey: capabilities.apiKey, presets: capabilities.presets, noModelsAutoload: capabilities.noModelsAutoload, modelsMax: capabilities.modelsMax, modelSleepIdleSeconds: capabilities.modelSleepIdleSeconds },
    devices: devices?._tag === "Right" ? devices.right.devices : [],
    ...(devices?._tag === "Left" ? { deviceProbeFailure: { operation: devices.left.operation, reason: devices.left.reason } } : {}),
  }
  yield* Schema.encode(JsonOutput)(report).pipe(Effect.flatMap(Console.log))
})

BunRuntime.runMain(program.pipe(Effect.provide(BunContext.layer)))
