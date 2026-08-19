import * as FetchHttpClient from "@effect/platform/FetchHttpClient"
import * as NodeCommandExecutor from "@effect/platform-node/NodeCommandExecutor"
import * as NodeFileSystem from "@effect/platform-node/NodeFileSystem"
import * as NodePath from "@effect/platform-node/NodePath"
import { NodeArchiveExtractor, RELAUNCH_EXIT_CODE } from "@magnitudedev/release"
import { Effect, Either, Layer, Option, Schema } from "effect"
import {
  cliBinaryResolverLayer,
  cliBinaryResolverPinnedLayer,
} from "./cli-binary-resolver"
import { CliProcessSpawner, cliProcessSpawnerLayer } from "./cli-process-spawner"
import {
  LauncherInstallationInspector,
  launcherInstallationInspectorLayer,
} from "./launcher-installation-inspector"

/**
 * The launcher entrypoint. Compiled by `scripts/build-launcher.ts` into the
 * published `bin/magnitude.js` (sh-polyglot banner prepended), so it must
 * locate the installation from process.argv[1] at runtime — bundling inlines
 * `__dirname` with the build machine's source path.
 */
const ProcessArgvSchema = Schema.Tuple(
  [Schema.String, Schema.String],
  Schema.String,
)

const decodedArgv = Schema.decodeUnknownEither(ProcessArgvSchema)(process.argv)
if (decodedArgv._tag === "Left") {
  console.error("Failed to launch Magnitude: launcher was started without an entrypoint path")
  process.exit(1)
}
const [, entrypoint, ...args] = decodedArgv.right
const environment = process.env
const pinnedBinary = environment.MAGNITUDE_CLI_BINARY

const platformLayer = Layer.mergeAll(
  NodeFileSystem.layer,
  NodePath.layer,
  FetchHttpClient.layer,
  NodeArchiveExtractor,
  NodeCommandExecutor.layer.pipe(Layer.provide(NodeFileSystem.layer)),
)
const inspectorLayer = launcherInstallationInspectorLayer({ entrypoint })
const resolverLayer = pinnedBinary !== undefined
  ? cliBinaryResolverPinnedLayer(pinnedBinary)
  : cliBinaryResolverLayer

const MainLive = Layer.mergeAll(
  cliProcessSpawnerLayer({ args, environment }).pipe(
    Layer.provide(resolverLayer),
    Layer.provide(inspectorLayer),
  ),
  inspectorLayer,
).pipe(Layer.provide(platformLayer))

// Each launch builds the layer graph fresh, so a relaunch re-inspects the
// installation (new package.json, new store paths) and resolves the newly
// installed binary.
const launchOnce = Effect.gen(function* () {
  const inspector = yield* LauncherInstallationInspector
  const spawner = yield* CliProcessSpawner
  const installation = yield* inspector.inspect
  const code = yield* spawner.spawn
  return { version: installation.version, code: Number(code) }
}).pipe(Effect.provide(MainLive))

const relaunchOnce = (previousVersion: string) => Effect.gen(function* () {
  const inspector = yield* LauncherInstallationInspector
  const spawner = yield* CliProcessSpawner
  const installation = yield* inspector.inspect
  // An unchanged version proves the update landed somewhere other than this
  // installation; running the old version again would only prompt again.
  if (installation.version === previousVersion) return Option.none<number>()
  return Option.some(Number(yield* spawner.spawn))
}).pipe(Effect.provide(MainLive))

const FLOOR_MESSAGE = "Update installed — run `magnitude` to start the new version."

const main = Effect.gen(function* () {
  const first = yield* Effect.either(launchOnce)
  if (Either.isLeft(first)) {
    console.error("Failed to launch Magnitude:", first.left.reason)
    return 1
  }
  if (first.right.code !== RELAUNCH_EXIT_CODE) return first.right.code
  // The CLI updated the installation and asked to be run again — honored at
  // most once; any failure or an unchanged installation degrades to a manual
  // restart.
  const second = yield* relaunchOnce(first.right.version).pipe(
    Effect.catchAll(() => Effect.sync(() => {
      console.error(FLOOR_MESSAGE)
      return Option.some(0)
    })),
  )
  return Option.match(second, {
    onNone: () => {
      console.error(FLOOR_MESSAGE)
      return 0
    },
    onSome: (code) => code,
  })
})

void Effect.runPromise(main).then(
  (code) => process.exit(code),
  (defect) => {
    console.error("Failed to launch Magnitude:", String(defect))
    process.exit(1)
  },
)
