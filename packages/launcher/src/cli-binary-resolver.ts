import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import * as Path from "@effect/platform/Path"
import { ArchiveExtractor } from "@magnitudedev/release"
import { ensureBinaryEffect } from "@magnitudedev/release/launcher"
import { Brand, Context, Effect, Layer, Schema } from "effect"
import {
  LauncherInstallationInspector,
  LauncherPackageNotFound,
} from "./launcher-installation-inspector"

/** A path to a runnable native CLI binary; minted only by resolver layers. */
export type ExecutablePath = string & Brand.Brand<"ExecutablePath">
const ExecutablePath = Brand.nominal<ExecutablePath>()

export class CliBinaryUnavailable extends Schema.TaggedError<CliBinaryUnavailable>()(
  "CliBinaryUnavailable",
  { reason: Schema.String },
) {}

/**
 * Resolves the installed version to a runnable native CLI binary on disk,
 * acquiring it from the release source when it is not already cached.
 */
export class CliBinaryResolver extends Context.Tag("launcher/CliBinaryResolver")<
  CliBinaryResolver, {
    readonly resolve: Effect.Effect<
      ExecutablePath,
      CliBinaryUnavailable | LauncherPackageNotFound
    >
  }
>() {}

export const cliBinaryResolverLayer: Layer.Layer<
  CliBinaryResolver,
  never,
  | LauncherInstallationInspector
  | CommandExecutor.CommandExecutor
  | FileSystem.FileSystem
  | HttpClient.HttpClient
  | Path.Path
  | ArchiveExtractor
> = Layer.effect(CliBinaryResolver, Effect.gen(function* () {
  const inspector = yield* LauncherInstallationInspector
  const context = yield* Effect.context<
    | CommandExecutor.CommandExecutor
    | FileSystem.FileSystem
    | HttpClient.HttpClient
    | Path.Path
    | ArchiveExtractor
  >()

  const resolve = Effect.gen(function* () {
    const installation = yield* inspector.inspect
    const binary = yield* ensureBinaryEffect(installation.version).pipe(
      Effect.provide(context),
      Effect.mapError((error) => new CliBinaryUnavailable({
        reason: error instanceof Error ? error.message : String(error),
      })),
    )
    return ExecutablePath(binary)
  })

  return { resolve }
}))

/** Dev mode: use a locally built native CLI binary instead of acquiring one. */
export const cliBinaryResolverPinnedLayer = (
  binary: string,
): Layer.Layer<CliBinaryResolver> => Layer.succeed(CliBinaryResolver, {
  resolve: Effect.succeed(ExecutablePath(binary)),
})
