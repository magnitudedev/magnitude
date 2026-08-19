import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import type { PackageManager } from "@magnitudedev/release"
import { Context, Effect, Layer, Option, Schema } from "effect"

export interface LauncherInstallation {
  /** Real path of the installed launcher package directory. */
  readonly root: string
  /** Version of the installed launcher package. */
  readonly version: string
  /** The package manager that owns the installation, when one is detectable. */
  readonly packageManager: Option.Option<PackageManager>
}

export class LauncherPackageNotFound extends Schema.TaggedError<LauncherPackageNotFound>()(
  "LauncherPackageNotFound",
  { reason: Schema.String },
) {}

/**
 * Inspects the launcher installation: where it is, what version it is, and
 * which package manager owns it.
 */
export class LauncherInstallationInspector extends Context.Tag(
  "launcher/LauncherInstallationInspector",
)<LauncherInstallationInspector, {
  readonly inspect: Effect.Effect<LauncherInstallation, LauncherPackageNotFound>
}>() {}

export interface LauncherInstallationInspectorConfig {
  /** The launcher entrypoint path as invoked (process.argv[1]), un-realpathed. */
  readonly entrypoint: string
}

const PackageManifestSchema = Schema.parseJson(Schema.Struct({
  version: Schema.String,
}))

const notFound = (reason: string) => new LauncherPackageNotFound({ reason })

export const launcherInstallationInspectorLayer = (
  config: LauncherInstallationInspectorConfig,
): Layer.Layer<
  LauncherInstallationInspector,
  never,
  FileSystem.FileSystem | Path.Path
> => Layer.effect(LauncherInstallationInspector, Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const path = yield* Path.Path

  const isPnpmOwnedInstall = (nodeModules: string, root: string) =>
    Effect.gen(function* () {
      const marker = yield* fs.exists(path.join(nodeModules, ".modules.yaml")).pipe(
        Effect.orElseSucceed(() => false),
      )
      if (!marker) return false
      return yield* fs.realPath(path.join(nodeModules, "@magnitudedev", "cli")).pipe(
        Effect.map((resolved) => resolved === root),
        Effect.orElseSucceed(() => false),
      )
    })

  // Detection reads only the filesystem — environment hints (user agent,
  // npm_execpath) describe whichever tool spawned this process, not whichever
  // one owns the installation, and any process launched from a package
  // manager's script context inherits them. pnpm and bun both leave positive
  // markers; npm leaves none in a global tree (verified: `npm install -g`
  // writes only the package directory), so with pnpm and bun ruled out no
  // evidence remains and the spawner's npm default carries the claim.
  const isBunOwnedInstall = (root: string) =>
    Effect.gen(function* () {
      const nodeModules = path.dirname(path.dirname(root))
      if (path.basename(nodeModules) !== "node_modules") return false
      const owner = path.dirname(nodeModules)
      for (const lockfile of ["bun.lock", "bun.lockb"]) {
        if (yield* fs.exists(path.join(owner, lockfile)).pipe(
          Effect.orElseSucceed(() => false),
        )) return true
      }
      return false
    })

  const detectPackageManager = (root: string) =>
    Effect.gen(function* () {
      const entrypointDirectory = path.dirname(path.resolve(config.entrypoint))
      for (const start of new Set([root, entrypointDirectory])) {
        const filesystemRoot = path.parse(start).root
        for (
          let current = start;
          ;
          current = path.dirname(current)
        ) {
          if (yield* isPnpmOwnedInstall(path.join(current, "node_modules"), root)) {
            return Option.some<PackageManager>("pnpm")
          }
          if (current === filesystemRoot) break
        }
      }

      if (yield* isBunOwnedInstall(root)) return Option.some<PackageManager>("bun")

      return Option.none<PackageManager>()
    })

  const inspect = Effect.gen(function* () {
    const entrypoint = yield* fs.realPath(path.resolve(config.entrypoint)).pipe(
      Effect.mapError(() => notFound(`launcher entrypoint ${config.entrypoint} is unreadable`)),
    )
    const root = path.dirname(path.dirname(entrypoint))
    const manifest = yield* fs.readFileString(path.join(root, "package.json")).pipe(
      Effect.mapError(() => notFound(`launcher package manifest is unreadable at ${root}`)),
    )
    const { version } = yield* Schema.decodeUnknown(PackageManifestSchema)(manifest).pipe(
      Effect.mapError(() => notFound("launcher package manifest has no version")),
    )
    const packageManager = yield* detectPackageManager(root)
    return { root, version, packageManager }
  })

  return { inspect: yield* Effect.cached(inspect) }
}))
