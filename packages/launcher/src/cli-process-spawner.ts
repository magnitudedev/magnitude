import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import {
  MANAGED_BY_VARIABLE,
  MANAGED_PACKAGE_ROOT_VARIABLE,
  type PackageManager,
} from "@magnitudedev/release"
import { Context, Effect, Layer, Option, Schema } from "effect"
import {
  CliBinaryResolver,
  type CliBinaryUnavailable,
} from "./cli-binary-resolver"
import {
  LauncherInstallationInspector,
  type LauncherInstallation,
  type LauncherPackageNotFound,
} from "./launcher-installation-inspector"

export class CliSpawnFailed extends Schema.TaggedError<CliSpawnFailed>()(
  "CliSpawnFailed",
  { reason: Schema.String },
) {}

/**
 * Spawns the native CLI on this terminal and reports its exit code. Encodes
 * install ownership into the child environment — the wire format of the
 * launcher→CLI contract — internally.
 */
export class CliProcessSpawner extends Context.Tag("launcher/CliProcessSpawner")<
  CliProcessSpawner, {
    readonly spawn: Effect.Effect<
      CommandExecutor.ExitCode,
      CliSpawnFailed | CliBinaryUnavailable | LauncherPackageNotFound
    >
  }
>() {}

export interface CliProcessSpawnerConfig {
  readonly args: ReadonlyArray<string>
  readonly environment: Readonly<Record<string, string | undefined>>
}

const definedEntries = (
  environment: Readonly<Record<string, string | undefined>>,
): Record<string, string> => {
  const defined: Record<string, string> = {}
  for (const [name, value] of Object.entries(environment)) {
    if (value !== undefined) defined[name] = value
  }
  return defined
}

const childEnvironment = (
  installation: LauncherInstallation,
  environment: Readonly<Record<string, string | undefined>>,
): Record<string, string> => ({
  ...definedEntries(environment),
  [MANAGED_BY_VARIABLE]: Option.getOrElse(
    installation.packageManager,
    (): PackageManager => "npm",
  ),
  [MANAGED_PACKAGE_ROOT_VARIABLE]: installation.root,
})

export const cliProcessSpawnerLayer = (
  config: CliProcessSpawnerConfig,
): Layer.Layer<
  CliProcessSpawner,
  never,
  | CliBinaryResolver
  | LauncherInstallationInspector
  | CommandExecutor.CommandExecutor
> => Layer.effect(CliProcessSpawner, Effect.gen(function* () {
  const resolver = yield* CliBinaryResolver
  const inspector = yield* LauncherInstallationInspector
  const executor = yield* CommandExecutor.CommandExecutor

  const spawn = Effect.gen(function* () {
    const installation = yield* inspector.inspect
    const binary = yield* resolver.resolve
    return yield* Command.make(binary, ...config.args).pipe(
      Command.env(childEnvironment(installation, config.environment)),
      Command.stdin("inherit"),
      Command.stdout("inherit"),
      Command.stderr("inherit"),
      Command.exitCode,
      Effect.provideService(CommandExecutor.CommandExecutor, executor),
      Effect.mapError((error) => new CliSpawnFailed({
        reason: error.message,
      })),
    )
  })

  return { spawn }
}))
