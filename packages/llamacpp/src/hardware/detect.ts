import { Context, Effect, Layer } from "effect"
import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import { BunFileSystem, BunCommandExecutor } from "@effect/platform-bun"
import { LlamaCppHardwareError } from "../errors"
import type {
  HardwareInfo,
  GpuDevice,
  AssessModelFitParams,
  ModelFitAssessment,
} from "./types"
import { getCpuInfo, getMemoryInfo } from "./os-info"
import { assessHeuristic, assessWithFitPrint } from "./fit"

const MIB = 1024 * 1024

const PlatformLayer = Layer.provideMerge(
  BunCommandExecutor.layer,
  BunFileSystem.layer,
)

export interface LlamaCppHardwareApi {
  readonly detect: () => Effect.Effect<HardwareInfo, LlamaCppHardwareError>

  readonly assessModelFit: (params: AssessModelFitParams) => Effect.Effect<
    ModelFitAssessment,
    LlamaCppHardwareError
  >
}

export class LlamaCppHardware extends Context.Tag("LlamaCppHardware")<
  LlamaCppHardware,
  LlamaCppHardwareApi
>() {}

export function makeLlamaCppHardware(deps: {
  readonly resolveBinaryPath: () => Effect.Effect<string, LlamaCppHardwareError>
}): LlamaCppHardwareApi {
  const detect: LlamaCppHardwareApi["detect"] = () =>
    Effect.gen(function* () {
      const binaryPath = yield* deps.resolveBinaryPath()
      const gpus = yield* listDevices(binaryPath)
      const cpu = getCpuInfo()
      const memory = yield* getMemoryInfo()

      const isUnifiedMemory =
        process.platform === "darwin" &&
        process.arch === "arm64" &&
        gpus.some((g) => g.backend.startsWith("MTL"))

      return { cpu, memory, gpus, isUnifiedMemory }
    }).pipe(
      Effect.mapError((err) =>
        err instanceof LlamaCppHardwareError
          ? err
          : new LlamaCppHardwareError({ reason: String(err) })
      ),
      Effect.provide(PlatformLayer),
    )

  const assessModelFit: LlamaCppHardwareApi["assessModelFit"] = (params) =>
    Effect.gen(function* () {
      if (params.modelPath) {
        const binaryPath = yield* deps.resolveBinaryPath()
        const precise = yield* assessWithFitPrint(
          binaryPath,
          params.modelPath,
          params.modelSizeBytes,
          params.hardware,
        ).pipe(Effect.catchAll(() => Effect.succeed(null)))

        if (precise) return precise
      }

      return assessHeuristic(params.hardware, params.modelSizeBytes)
    }).pipe(
      Effect.mapError((err) =>
        err instanceof LlamaCppHardwareError
          ? err
          : new LlamaCppHardwareError({ reason: String(err) })
      ),
      Effect.provide(PlatformLayer),
    )

  return { detect, assessModelFit }
}

function listDevices(
  binaryPath: string,
): Effect.Effect<readonly GpuDevice[], LlamaCppHardwareError, CommandExecutor.CommandExecutor> {
  return Effect.gen(function* () {
    const result = yield* Command.string(
      Command.make(binaryPath, "--list-devices"),
    ).pipe(
      Effect.mapError((cause) =>
        new LlamaCppHardwareError({ reason: `--list-devices failed: ${cause}` }),
      ),
    )
    return parseListDevicesOutput(result)
  })
}

export function parseListDevicesOutput(output: string): readonly GpuDevice[] {
  const lines = output.split("\n").map((l) => l.trim()).filter((l) => l.length > 0)
  const devices: GpuDevice[] = []

  for (const line of lines) {
    const match = line.match(/^(\w+):\s+(.+?)\s+\((\d+)\s*MiB,\s*(\d+)\s*MiB\s+free\)/)
    if (!match) continue

    const [, backend, name, totalMiB, freeMiB] = match
    const totalBytes = Number(totalMiB) * MIB
    const freeBytes = Number(freeMiB) * MIB

    if (totalBytes === 0) continue

    devices.push({
      backend: backend!,
      name: name!,
      totalBytes,
      freeBytes,
    })
  }

  return devices
}
