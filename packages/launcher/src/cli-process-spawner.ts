import { interactiveProcessExitCode, runInteractiveProcess } from "@magnitudedev/utils/process"
import { Context, Effect, Layer, Schema } from "effect"
import { CliBinaryResolver, type CliBinaryUnavailable } from "./cli-binary-resolver"

export class CliSpawnFailed extends Schema.TaggedError<CliSpawnFailed>()("CliSpawnFailed", { reason: Schema.String }) {}

export interface CliProcessSpawner {
  readonly spawn: Effect.Effect<number, CliSpawnFailed | CliBinaryUnavailable>
}
export const CliProcessSpawner = Context.GenericTag<CliProcessSpawner>("launcher/CliProcessSpawner")

export const cliProcessSpawnerLayer = (config: {
  readonly args: ReadonlyArray<string>
  readonly environment: Readonly<Record<string, string | undefined>>
}) => Layer.effect(CliProcessSpawner, Effect.gen(function* () {
  const resolver = yield* CliBinaryResolver
  return { spawn: Effect.gen(function* () {
    const binary = yield* resolver.resolve
    const environment: Record<string, string> = {}
    for (const [name, value] of Object.entries(config.environment)) {
      if (value !== undefined) environment[name] = value
    }
    environment.MAGNITUDE_DESKTOP_PATH = binary.application
    return yield* runInteractiveProcess({ executable: binary.executable, args: config.args, environment }).pipe(
      Effect.map(interactiveProcessExitCode),
      Effect.mapError(error => new CliSpawnFailed({ reason: error.message })),
    )
  }) }
}))
