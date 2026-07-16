import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import { Data, Effect, Option, Redacted, Schema, Scope } from "effect"
import type { ResolvedModelFiles } from "../model-files"
import { CommandCaptureOptions, captureProcessOutput, runParsedCommand, type CommandOutputParser } from "../command-output"
import { LlamaCliError, type LlamaCliOperation } from "./cli-errors"
import { renderExecutionProfileArguments, type LlamaExecutionProfile } from "./execution-profile"
import { boundedFitDiagnostic, LlamaFitResult, makeLlamaFitPlan, parseLlamaFitPlacement, type LlamaFitResult as LlamaFitResultType } from "./fit"
import { LlamaDeviceId } from "./identity"
import { LlamaBinary, type LlamaBuildIdentity } from "./binary"

export const LlamaRouterHost = Schema.Literal("127.0.0.1", "::1")
export type LlamaRouterHost = Schema.Schema.Type<typeof LlamaRouterHost>
export const LlamaStartOperation = Schema.Literal("capabilities", "validate-options", "start", "observe-exit")
export type LlamaStartOperation = Schema.Schema.Type<typeof LlamaStartOperation>
export const LlamaStartFailureReason = Schema.Literal("command-failed", "missing-capability", "invalid-input")
export type LlamaStartFailureReason = Schema.Schema.Type<typeof LlamaStartFailureReason>
export interface LlamaCliCapabilities {
  readonly fitPrint: boolean
  readonly listDevices: boolean
  readonly apiKey: boolean
  readonly presets: boolean
  readonly noModelsAutoload: boolean
  readonly modelsMax: boolean
  readonly modelSleepIdleSeconds: boolean
  readonly helpOutput: string
}

export interface LlamaDevice {
  readonly id: Schema.Schema.Type<typeof LlamaDeviceId>
  readonly name: Option.Option<string>
  readonly type: Option.Option<string>
  readonly totalMemoryBytes: Option.Option<number>
  readonly freeMemoryBytes: Option.Option<number>
}

export interface LlamaDeviceSnapshot {
  readonly binaryFingerprint: LlamaBinary["fingerprint"]
  readonly capturedAt: Date
  readonly devices: readonly LlamaDevice[]
  readonly rawOutput: string
}

export interface LlamaRouterOptions {
  readonly presetPath: string
  readonly host: LlamaRouterHost
  readonly port: number
  readonly apiKey: Redacted.Redacted<string>
  readonly modelsMax: Option.Option<number>
  readonly modelSleepIdleSeconds: Option.Option<number>
}

export interface RunningLlamaProcess {
  readonly origin: URL
  readonly exited: Effect.Effect<number, LlamaStartError>
  readonly sanitizedOutput: Effect.Effect<string>
}
export class LlamaStartError extends Data.TaggedError("LlamaStartError")<{ readonly operation: LlamaStartOperation; readonly reason: LlamaStartFailureReason; readonly capability: Option.Option<keyof Omit<LlamaCliCapabilities, "helpOutput">> }> {}
export interface LlamaFitAssessmentInput {
  readonly files: ResolvedModelFiles
  readonly profile: LlamaExecutionProfile
}

export interface LlamaCli {
  readonly binary: LlamaBinary
  readonly version: Effect.Effect<LlamaBuildIdentity>
  readonly capabilities: Effect.Effect<LlamaCliCapabilities, LlamaCliError>
  readonly listDevices: Effect.Effect<LlamaDeviceSnapshot, LlamaCliError>
  readonly assessFit: (
    input: LlamaFitAssessmentInput,
  ) => Effect.Effect<LlamaFitResultType, LlamaCliError>
  readonly startRouter: (
    options: LlamaRouterOptions,
  ) => Effect.Effect<RunningLlamaProcess, LlamaStartError, Scope.Scope>
}

const DeviceJson = Schema.Struct({ id: LlamaDeviceId, name: Schema.optional(Schema.String), type: Schema.optional(Schema.String), total_memory: Schema.optional(Schema.NonNegativeInt), free_memory: Schema.optional(Schema.NonNegativeInt) })
const DeviceListJson = Schema.parseJson(Schema.Array(DeviceJson))
const cliError = (operation: LlamaCliOperation, reason: LlamaCliError["reason"]) => LlamaCliError.make(operation, reason)
const startError = (operation: LlamaStartOperation, reason: LlamaStartFailureReason) => new LlamaStartError({ operation, reason, capability: Option.none() })

const hasFlag = (help: string, flag: string): boolean => new RegExp(`(^|[\\s,])${flag.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?=[\\s,=]|$)`, "m").test(help)
const documentedDevices = (output: string): Effect.Effect<readonly LlamaDevice[], LlamaCliError> => Effect.gen(function* () {
  const devices: LlamaDevice[] = []
  for (const line of output.split("\n")) {
    const match = Option.fromNullable(line.match(/^\s{2}([^:]+):\s+(.+?)\s+\((\d+) MiB,\s*(\d+) MiB free\)$/))
    if (Option.isNone(match)) continue
    const values = Option.all({
      rawId: Option.fromNullable(match.value[1]),
      name: Option.fromNullable(match.value[2]),
      totalMiB: Option.fromNullable(match.value[3]),
      freeMiB: Option.fromNullable(match.value[4]),
    })
    if (Option.isNone(values)) return yield* LlamaCliError.make("list-devices", "invalid-output")
    const id = yield* Schema.decodeUnknown(LlamaDeviceId)(values.value.rawId).pipe(Effect.mapError(() => LlamaCliError.make("list-devices", "invalid-output")))
    devices.push({ id, name: Option.some(values.value.name), type: Option.none(), totalMemoryBytes: Option.some(Number(values.value.totalMiB) * 1024 * 1024), freeMemoryBytes: Option.some(Number(values.value.freeMiB) * 1024 * 1024) })
  }
  if (devices.length === 0) return yield* new LlamaCliError({ operation: "list-devices", reason: "invalid-output", field: Option.none() })
  return devices
})

export const makeLlamaCli = (): Effect.Effect<
  LlamaCli,
  never,
  LlamaBinary | CommandExecutor.CommandExecutor
> => Effect.gen(function* () {
  const binary = yield* LlamaBinary
  const executor = yield* CommandExecutor.CommandExecutor
  const output = (operation: LlamaCliError["operation"], args: readonly string[]) => {
    const parser: CommandOutputParser<string, LlamaCliError> = {
      name: `llama-${operation}`,
      parse: (captured) => captured.exitCode === 0
        ? Effect.succeed(`${captured.stdout}\n${captured.stderr}`.trim())
        : Effect.fail(cliError(operation, "command-failed")),
    }
    return runParsedCommand(binary.executable, args, parser, CommandCaptureOptions.Default).pipe(Effect.provideService(CommandExecutor.CommandExecutor, executor), Effect.mapError((error) => error._tag === "LlamaCliError" ? error : cliError(operation, "command-failed")))
  }
  const capabilities = yield* output("help", ["--help"]).pipe(
    Effect.map((helpOutput) => ({ fitPrint: hasFlag(helpOutput, "--fit-print"), listDevices: hasFlag(helpOutput, "--list-devices"), apiKey: hasFlag(helpOutput, "--api-key"), presets: hasFlag(helpOutput, "--models-preset"), noModelsAutoload: hasFlag(helpOutput, "--no-models-autoload"), modelsMax: hasFlag(helpOutput, "--models-max"), modelSleepIdleSeconds: hasFlag(helpOutput, "--sleep-idle-seconds"), helpOutput })),
    Effect.cached,
  )
  return {
    binary,
    version: Effect.succeed(binary.build),
    capabilities,
    listDevices: Effect.gen(function* () {
      if (!(yield* capabilities).listDevices) return yield* cliError("list-devices", "unsupported")
      const rawOutput = yield* output("list-devices", ["--list-devices"])
      const decoded = yield* Schema.decode(DeviceListJson)(rawOutput).pipe(Effect.option)
      const devices = Option.isSome(decoded)
        ? decoded.value.map((item) => ({ id: item.id, name: Option.fromNullable(item.name), type: Option.fromNullable(item.type), totalMemoryBytes: Option.fromNullable(item.total_memory), freeMemoryBytes: Option.fromNullable(item.free_memory) }))
        : yield* documentedDevices(rawOutput)
      return { binaryFingerprint: binary.fingerprint, capturedAt: new Date(), devices, rawOutput }
    }),
    assessFit: ({ files, profile }) => Effect.gen(function* () {
      if (!(yield* capabilities).fitPrint) return LlamaFitResult.Unsupported({ binary: binary.fingerprint })
      const projectorArguments = Option.match(files.projectorPath, {
        onNone: () => [],
        onSome: (projectorPath) => ["--mmproj", projectorPath],
      })
      const args = ["--model", files.primaryPath, ...files.shardPaths.flatMap((file) => ["--model", file]), ...projectorArguments, ...renderExecutionProfileArguments(profile), "--fit-print", "on"]
      const rawOutput = yield* output("fit", args)
      const placement = parseLlamaFitPlacement(rawOutput)
      if (Option.isNone(placement)) return LlamaFitResult.InvalidOutput({ diagnostic: boundedFitDiagnostic(rawOutput) })
      return LlamaFitResult.Measured({ plan: makeLlamaFitPlan({ binaryFingerprint: binary.fingerprint, profileId: profile.id, fileVersion: files.version, arguments: args, placement: placement.value, rawOutput }) })
    }),
    startRouter: (options) => Effect.gen(function* () {
      const caps = yield* capabilities.pipe(Effect.mapError(() => startError("capabilities", "command-failed")))
      const missing = Option.fromNullable((["apiKey", "presets", "noModelsAutoload"] as const).find((capability) => !caps[capability]))
      if (Option.isSome(missing)) return yield* new LlamaStartError({ operation: "validate-options", reason: "missing-capability", capability: missing })
      if (!Number.isSafeInteger(options.port) || options.port <= 0 || options.port > 65535) return yield* startError("validate-options", "invalid-input")
      if (Option.isSome(options.modelsMax) && !caps.modelsMax) return yield* new LlamaStartError({ operation: "validate-options", reason: "missing-capability", capability: Option.some("modelsMax") })
      if (Option.isSome(options.modelSleepIdleSeconds) && !caps.modelSleepIdleSeconds) return yield* new LlamaStartError({ operation: "validate-options", reason: "missing-capability", capability: Option.some("modelSleepIdleSeconds") })
      const args = ["--host", options.host, "--port", String(options.port), "--api-key", Redacted.value(options.apiKey), "--models-preset", options.presetPath, "--no-models-autoload"]
      Option.map(options.modelsMax, (modelsMax) => args.push("--models-max", String(modelsMax)))
      Option.map(options.modelSleepIdleSeconds, (seconds) => args.push("--sleep-idle-seconds", String(seconds)))
      const process = yield* Command.start(Command.make(binary.executable, ...args)).pipe(Effect.provideService(CommandExecutor.CommandExecutor, executor), Effect.mapError(() => startError("start", "command-failed")))
      const capture = yield* captureProcessOutput(binary.executable, process, { maxOutputBytes: CommandCaptureOptions.ProcessDefault.maxOutputBytes, redactions: [options.apiKey] })
      const address = options.host === "::1" ? "[::1]" : options.host
      return {
        origin: new URL(`http://${address}:${options.port}`),
        exited: process.exitCode.pipe(
          Effect.tap(() => capture.completed),
          Effect.map(Number),
          Effect.mapError(() => startError("observe-exit", "command-failed")),
        ),
        sanitizedOutput: capture.text,
      }
    }),
  }
})
