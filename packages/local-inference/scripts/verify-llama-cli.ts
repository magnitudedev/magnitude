import * as BunContext from "@effect/platform-bun/BunContext"
import * as BunRuntime from "@effect/platform-bun/BunRuntime"
import { Console, Data, Effect, Schema } from "effect"
import { makeLlamaCli, makeLlamaCppInstallation, validateLlamaCppExecutable } from "../src/llamacpp"

class MissingExecutable extends Data.TaggedError("MissingExecutable")<{}> {}
const JsonOutput = Schema.parseJson(Schema.Unknown, { space: 2 })

const program = Effect.gen(function* () {
  const serverExecutable = process.argv[2]
  const fitParamsExecutable = process.argv[3]
  if (serverExecutable === undefined || fitParamsExecutable === undefined) return yield* new MissingExecutable()
  const installation = yield* Effect.all({
    server: validateLlamaCppExecutable(serverExecutable),
    fitParams: validateLlamaCppExecutable(fitParamsExecutable),
  }, { concurrency: 2 }).pipe(Effect.flatMap((executables) => makeLlamaCppInstallation({
    ...executables,
    ownership: "user",
    discoveries: [{ _tag: "Configured", requestedPath: serverExecutable }],
  })))
  const cli = yield* makeLlamaCli(installation)
  const devices = yield* cli.listDevices.pipe(Effect.either)
  const report = {
    verifiedAt: new Date().toISOString(), installation,
    devices: devices?._tag === "Right" ? devices.right.devices : [],
    ...(devices?._tag === "Left" ? { deviceProbeFailure: { operation: devices.left.operation, reason: devices.left.reason } } : {}),
  }
  yield* Schema.encode(JsonOutput)(report).pipe(Effect.flatMap(Console.log))
})

BunRuntime.runMain(program.pipe(Effect.provide(BunContext.layer)))
