import * as CommandExecutor from "@effect/platform/CommandExecutor"
import { arch, availableParallelism, cpus, freemem, platform, totalmem } from "node:os"
import { Context, Data, Effect, Option, pipe } from "effect"
import { CommandCaptureOptions, runParsedCommand, type CommandOutputParser } from "../command-output"

export interface HostHardwareSnapshot {
  readonly capturedAt: Date
  readonly platform: NodeJS.Platform
  readonly processArchitecture: string
  readonly nativeArchitecture: string
  readonly cpuModel: Option.Option<string>
  readonly logicalCores: number
  readonly totalMemoryBytes: number
  readonly availableMemoryBytes: number
}

export class NativeArchitectureCommandError extends Data.TaggedError("NativeArchitectureCommandError")<{
  readonly platform: NodeJS.Platform
}> {}

export class NativeArchitectureOutputError extends Data.TaggedError("NativeArchitectureOutputError")<{
  readonly platform: NodeJS.Platform
  readonly output: string
}> {}

export type HostHardwareError = NativeArchitectureCommandError | NativeArchitectureOutputError

export interface HostHardwareApi {
  readonly inspect: Effect.Effect<HostHardwareSnapshot, HostHardwareError>
}

export class HostHardware extends Context.Tag("@magnitudedev/local-inference/HostHardware")<HostHardware, HostHardwareApi>() {}

const commandForNativeArchitecture = (
  hostPlatform: NodeJS.Platform,
): { readonly executable: string; readonly arguments: readonly string[] } => {
  if (hostPlatform === "darwin") {
    return {
      executable: "/usr/sbin/sysctl",
      arguments: ["-n", "hw.optional.arm64"],
    }
  }

  return {
    executable: "uname",
    arguments: ["-m"],
  }
}

const parseNativeArchitecture = (
  hostPlatform: NodeJS.Platform,
  processArchitecture: string,
  value: string,
): Effect.Effect<string, NativeArchitectureOutputError> => {
  const normalize = (architecture: string): string => {
    const value = architecture.trim().toLowerCase()
    if (value === "x86_64" || value === "amd64" || value === "x64") return "x64"
    if (value === "aarch64" || value === "arm64") return "arm64"
    return architecture.trim()
  }
  if (hostPlatform !== "darwin") {
    return value.length > 0
      ? Effect.succeed(normalize(value))
      : Effect.fail(new NativeArchitectureOutputError({ platform: hostPlatform, output: value }))
  }

  if (value === "1") return Effect.succeed("arm64")
  if (value === "0") return Effect.succeed(normalize(processArchitecture))

  return Effect.fail(new NativeArchitectureOutputError({
    platform: hostPlatform,
    output: value,
  }))
}

const inspectNativeArchitecture = (hostPlatform: NodeJS.Platform, processArchitecture: string): Effect.Effect<string, HostHardwareError, CommandExecutor.CommandExecutor> => {
  if (hostPlatform === "win32") {
    const native = Option.orElse(
      Option.fromNullable(process.env.PROCESSOR_ARCHITEW6432),
      () => Option.fromNullable(process.env.PROCESSOR_ARCHITECTURE),
    )
    return Effect.succeed(Option.getOrElse(native, () => processArchitecture)).pipe(
      Effect.flatMap((value) => parseNativeArchitecture("win32", processArchitecture, value)),
    )
  }
  const command = commandForNativeArchitecture(hostPlatform)
  const parser: CommandOutputParser<string, HostHardwareError> = {
    name: "native-architecture",
    parse: (output) => {
      if (output.exitCode !== 0) {
        return Effect.fail(new NativeArchitectureCommandError({ platform: hostPlatform }))
      }

      const value = output.stdout.trim()
      return parseNativeArchitecture(hostPlatform, processArchitecture, value)
    },
  }

  return runParsedCommand(
    command.executable,
    command.arguments,
    parser,
    CommandCaptureOptions.Default,
  ).pipe(
    Effect.mapError((error) => error._tag === "CommandExecutionError"
      ? new NativeArchitectureCommandError({ platform: hostPlatform })
      : error),
  )
}

export const makeHostHardware = (): Effect.Effect<HostHardwareApi, never, CommandExecutor.CommandExecutor> => Effect.gen(function* () {
  const executor = yield* CommandExecutor.CommandExecutor

  const inspect = Effect.gen(function* () {
    const hostPlatform = platform()
    const processArchitecture = arch()
    const cpuList = cpus()
    const nativeArchitecture = yield* inspectNativeArchitecture(
      hostPlatform,
      processArchitecture,
    ).pipe(Effect.provideService(CommandExecutor.CommandExecutor, executor))

    return {
      capturedAt: new Date(),
      platform: hostPlatform,
      processArchitecture,
      nativeArchitecture,
      cpuModel: pipe(
        Option.fromNullable(cpuList[0]),
        Option.map(({ model }) => model),
      ),
      logicalCores: cpuList.length > 0 ? cpuList.length : availableParallelism(),
      totalMemoryBytes: totalmem(),
      availableMemoryBytes: freemem(),
    }
  })

  return { inspect }
})
