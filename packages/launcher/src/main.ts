import * as FetchHttpClient from "@effect/platform/FetchHttpClient"
import * as NodeCommandExecutor from "@effect/platform-node/NodeCommandExecutor"
import * as NodeFileSystem from "@effect/platform-node/NodeFileSystem"
import * as NodePath from "@effect/platform-node/NodePath"
import { NodeArchiveExtractor } from "@magnitudedev/release"
import { Effect, Layer, Schema } from "effect"
import {
  cliBinaryResolverLayer,
  cliBinaryResolverPinnedLayer,
} from "./cli-binary-resolver"
import { CliProcessSpawner, cliProcessSpawnerLayer } from "./cli-process-spawner"
import { launcherInstallationInspectorLayer } from "./launcher-installation-inspector"

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
const inspectorLayer = launcherInstallationInspectorLayer({ entrypoint, environment })
const resolverLayer = pinnedBinary !== undefined
  ? cliBinaryResolverPinnedLayer(pinnedBinary)
  : cliBinaryResolverLayer

const MainLive = cliProcessSpawnerLayer({ args, environment }).pipe(
  Layer.provide(resolverLayer),
  Layer.provide(inspectorLayer),
  Layer.provide(platformLayer),
)

const main = CliProcessSpawner.pipe(
  Effect.flatMap((spawner) => spawner.spawn),
  Effect.provide(MainLive),
  Effect.catchAll((error) => Effect.sync(() => {
    console.error("Failed to launch Magnitude:", error.reason)
    return 1
  })),
)

void Effect.runPromise(main).then(
  (code) => process.exit(code),
  (defect) => {
    console.error("Failed to launch Magnitude:", String(defect))
    process.exit(1)
  },
)
