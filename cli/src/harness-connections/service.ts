import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import * as Path from "@effect/platform/Path"
import {
  HarnessConnectionError,
  HarnessIdSchema,
  type HarnessConnection,
  type HarnessDestination,
  type HarnessId,
} from "@magnitudedev/client-common"
import { ProviderModelIdSchema, type ProviderModelId } from "@magnitudedev/sdk"
import { makeStateDocument } from "@magnitudedev/storage"
import { Data, Effect, Option, Schema } from "effect"
import { delimiter } from "node:path"
import { installServiceOnStartup } from "../server/service"
import { writeFileAtomic } from "../utils/atomic-file"
import {
  HarnessModelSchema,
  type HarnessConnectionSpec,
  type HarnessConnector,
  type HarnessInstallation,
  type HarnessModel,
} from "./contract"
import { harnessConnectionPaths, type HarnessConnectionPaths } from "./paths"
import { makeHarnessConnectorRegistry } from "./registry"
import { OPENAI_BASE_URL, readOr } from "./shared"

export {
  ANTHROPIC_BASE_URL,
  OPENAI_BASE_URL,
  anthropicLocalModelId,
  removeOwnedJsonc,
  updateJsonc,
  updateYaml,
} from "./shared"
export { clineModelCatalog, clineModelRegistryEntry, clineProviderSettings } from "./connectors/cline"
export { CLAUDE_GATEWAY_DISCOVERY } from "./connectors/claude-code"
export { codexConfig, codexModelCatalog } from "./connectors/codex"
export { hermesProviderConfig } from "./connectors/hermes"
export { ohMyPiProviderConfig } from "./connectors/oh-my-pi"
export { openClawAgentConfig, openClawProviderConfig } from "./connectors/openclaw"
export { openCodeProviderConfig } from "./connectors/opencode"
export { piProviderConfig } from "./connectors/pi"
export { makeHarnessConnectorRegistry } from "./registry"
export { harnessConnectionPaths, type HarnessConnectionPaths } from "./paths"
export type { HarnessConnectionSpec, HarnessConnector, HarnessInstallation, HarnessModel } from "./contract"

const ManifestEntrySchema = Schema.Struct({
  harness: HarnessIdSchema,
  models: Schema.optionalWith(Schema.Array(HarnessModelSchema), { default: () => [] }),
  setCurrent: Schema.optionalWith(ProviderModelIdSchema, { as: "Option", exact: true }),
  updatedAt: Schema.optionalWith(Schema.String, { default: () => "1970-01-01T00:00:00.000Z" }),
})
const ManifestSchema = Schema.Struct({
  connections: Schema.Array(ManifestEntrySchema),
})
type Manifest = typeof ManifestSchema.Type
type ManifestEntry = typeof ManifestEntrySchema.Type

const emptyManifest = (): Manifest => ({ connections: [] })

const failure = (
  operation: HarnessConnectionError["operation"],
  message: string,
  harness?: HarnessId,
) => new HarnessConnectionError({ operation, harness, message })

export const harnessExecutableSearchPath = (value = process.env.PATH ?? ""): string => value
  .split(delimiter)
  .filter((directory) => !directory.replaceAll("\\", "/").endsWith("/node_modules/.bin"))
  .join(delimiter)

export interface HarnessConnectionOptions {
  readonly paths?: HarnessConnectionPaths
  readonly detect?: (connector: HarnessConnector) => Effect.Effect<Option.Option<HarnessInstallation>>
  readonly resolveModels?: Effect.Effect<ReadonlyArray<HarnessModel>, unknown, HttpClient.HttpClient>
  readonly now?: () => Date
  readonly installStartup?: Effect.Effect<
    void,
    unknown,
    FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor
  >
}

const upsertManifest = (
  manifest: Manifest,
  harness: HarnessId,
  spec: HarnessConnectionSpec,
  now: () => Date,
) => ({
  connections: [
    ...manifest.connections.filter((entry) => entry.harness !== harness),
    { harness, models: [...spec.models], setCurrent: spec.setCurrent, updatedAt: now().toISOString() },
  ],
}) satisfies Manifest

const entrySpec = (entry: ManifestEntry): HarnessConnectionSpec => ({
  models: entry.models,
  setCurrent: entry.setCurrent,
})

const ModelsResponseSchema = Schema.Struct({
  models: Schema.Array(HarnessModelSchema),
})

class HarnessModelDiscoveryFailed extends Data.TaggedError("HarnessModelDiscoveryFailed")<{
  readonly message: string
}> {}

const discoverMagnitudeModels = Effect.gen(function* () {
  const http = yield* HttpClient.HttpClient
  const response = yield* http.execute(HttpClientRequest.get(`${OPENAI_BASE_URL}/models`))
  if (response.status < 200 || response.status >= 300) {
    return yield* new HarnessModelDiscoveryFailed({
      message: `Magnitude model discovery returned HTTP ${response.status}`,
    })
  }
  const decoded = yield* response.json.pipe(Effect.flatMap(Schema.decodeUnknown(ModelsResponseSchema)))
  return decoded.models
})

const uniqueModels = (models: ReadonlyArray<HarnessModel>): ReadonlyArray<HarnessModel> => models.filter(
  (model, index) => models.findIndex((candidate) => candidate.id === model.id) === index,
)

const containsModel = (models: ReadonlyArray<HarnessModel>, modelId: ProviderModelId): boolean =>
  models.some(({ id }) => id === modelId)

const skillContents = `---
name: magnitude
description: Instructions for \`magnitude\` cli to discover, install, and manage local models, as well as connect local models to different agents
---

# Magnitude

Magnitude is a local inference service with a built-in harness for discovering, managing, and using local models.

Magnitude is accessed via the \`magnitude\` CLI.

## CLI Help

Use \`magnitude --help\` to get more information on using the CLI.
`

export const makeHarnessConnectionService = (options: HarnessConnectionOptions = {}) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const runtime = yield* Effect.runtime<
    FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor | HttpClient.HttpClient
  >()
  const paths = options.paths ?? harnessConnectionPaths()
  const registry = makeHarnessConnectorRegistry(paths)
  const now = options.now ?? (() => new Date())
  const manifestState = yield* makeStateDocument({
    path: paths.manifest,
    schema: ManifestSchema,
    initial: emptyManifest,
    equivalence: Schema.equivalence(ManifestSchema),
  })
  const readManifest = manifestState.get
  const updateManifest = manifestState.update
  const detect = options.detect ?? ((connector: HarnessConnector) => connector.detect(harnessExecutableSearchPath()))
  const resolveModels = options.resolveModels ?? discoverMagnitudeModels
  const mutationLock = yield* Effect.makeSemaphore(1)
  const provide = <A, E>(effect: Effect.Effect<
    A,
    E,
    FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor | HttpClient.HttpClient
  >) => Effect.provide(effect, runtime)

  const snapshotFiles = (files: ReadonlyArray<string>) => Effect.forEach(
    files,
    (file) => fs.readFileString(file).pipe(
      Effect.map((contents) => ({ file, contents: Option.some(contents) })),
      Effect.catchTag("SystemError", (error) => error.reason === "NotFound"
        ? Effect.succeed({ file, contents: Option.none<string>() })
        : Effect.fail(error)),
    ),
  )

  const restoreFiles = (
    snapshots: ReadonlyArray<{ readonly file: string; readonly contents: Option.Option<string> }>,
  ) => Effect.forEach(snapshots, ({ file, contents }) => Option.match(contents, {
    onSome: (source) => writeFileAtomic(file, source),
    onNone: () => fs.remove(file).pipe(
      Effect.catchTag("SystemError", (error) => error.reason === "NotFound" ? Effect.void : Effect.fail(error)),
    ),
  }), { discard: true })

  const connectorOperation = (
    operation: HarnessConnectionError["operation"],
    connector: HarnessConnector,
    effect: Effect.Effect<void, unknown, FileSystem.FileSystem | Path.Path>,
  ) => effect.pipe(
    Effect.mapError((error) => error instanceof HarnessConnectionError
      ? error
      : failure(operation, String(error), connector.id)),
    Effect.catchAllDefect((defect) => Effect.fail(failure(
      operation,
      defect instanceof Error ? defect.message : String(defect),
      connector.id,
    ))),
  )

  const installed = (connector: HarnessConnector) => provide(detect(connector))

  const list: HarnessConnection["list"] = provide(Effect.forEach(registry.ordered, (connector) => installed(connector).pipe(
    Effect.map((installation): HarnessDestination => ({
      id: connector.id,
      name: connector.name,
      availability: Option.isSome(installation) ? "Installed" : "Not installed",
      selectable: connector.recommended || Option.isSome(installation),
      ...(connector.note === undefined ? {} : { note: connector.note }),
    })),
  )).pipe(
    Effect.map((rows) => [...rows.filter((row) => row.selectable), ...rows.filter((row) => !row.selectable)]),
    Effect.mapError((error) => failure("list", String(error))),
  ))

  const installStartup = provide(options.installStartup ?? installServiceOnStartup).pipe(
    Effect.mapError((error) => failure("startup", String(error))),
  )

  const withMutationLock = <A, E, R>(effect: Effect.Effect<A, E, R>) =>
    mutationLock.withPermits(1)(effect)

  const connect: HarnessConnection["connect"] = (harness, connectOptions) => provide(Effect.gen(function* () {
    const connector = registry.get(harness)
    const installation = yield* installed(connector)
    if (Option.isNone(installation)) return yield* failure("connect", `${connector.name} is not installed`, harness)
    const models = uniqueModels(yield* resolveModels)
    if (models.length === 0) return yield* failure("connect", "No installed Magnitude models are available", harness)
    if (Option.isSome(connectOptions.setCurrent) && !containsModel(models, connectOptions.setCurrent.value)) {
      return yield* failure("connect", `Magnitude model is not installed: ${connectOptions.setCurrent.value}`, harness)
    }
    const spec: HarnessConnectionSpec = { models, setCurrent: connectOptions.setCurrent }
    if (connector.recommended) {
      return { launchPlan: Option.map(spec.setCurrent, (modelId) => connector.launch(modelId, installation.value)) }
    }
    if (connector.requiresStartup) yield* installStartup
    const snapshots = yield* snapshotFiles(connector.configurationFiles)
    yield* connectorOperation("connect", connector, connector.connect(spec)).pipe(
      Effect.zipRight(updateManifest((manifest) => upsertManifest(manifest, harness, spec, now))),
      Effect.onError(() => restoreFiles(snapshots).pipe(Effect.ignore)),
    )
    return { launchPlan: Option.map(spec.setCurrent, (modelId) => connector.launch(modelId, installation.value)) }
  }).pipe(Effect.mapError((error) => error instanceof HarnessConnectionError
    ? error
    : failure("connect", String(error), harness))))

  const sync: HarnessConnection["sync"] = (harness) => provide(Effect.gen(function* () {
    let manifest = yield* readManifest
    const entries = harness === undefined
      ? manifest.connections
      : manifest.connections.filter((entry) => entry.harness === harness)
    if (harness !== undefined && entries.length === 0) {
      return yield* failure("sync", `${registry.get(harness).name} has no Magnitude harness connection`, harness)
    }
    const models = uniqueModels(yield* resolveModels)
    if (models.length === 0) return yield* failure("sync", "No installed Magnitude models are available", harness)
    for (const entry of entries) {
      const connector = registry.get(entry.harness)
      if (Option.isSome(entry.setCurrent) && !containsModel(models, entry.setCurrent.value)) {
        return yield* failure("sync", `Magnitude model is not installed: ${entry.setCurrent.value}`, entry.harness)
      }
      const spec: HarnessConnectionSpec = { models, setCurrent: entry.setCurrent }
      if (connector.requiresStartup) yield* installStartup
      const snapshots = yield* snapshotFiles(connector.configurationFiles)
      yield* connectorOperation("sync", connector, connector.connect(spec)).pipe(
        Effect.zipRight(updateManifest((current) => upsertManifest(current, entry.harness, spec, now))),
        Effect.onError(() => restoreFiles(snapshots).pipe(Effect.ignore)),
      )
      manifest = yield* readManifest
    }
    return yield* list
  }).pipe(Effect.mapError((error) => error instanceof HarnessConnectionError
    ? error
    : failure("sync", String(error), harness))))

  const disconnect: HarnessConnection["disconnect"] = (harness) => provide(Effect.gen(function* () {
    const manifest = yield* readManifest
    const entry = manifest.connections.find((candidate) => candidate.harness === harness)
    if (entry === undefined) return
    const connector = registry.get(harness)
    const snapshots = yield* snapshotFiles(connector.configurationFiles)
    yield* connectorOperation("disconnect", connector, connector.disconnect(entrySpec(entry))).pipe(
      Effect.zipRight(updateManifest((current) => ({
        connections: current.connections.filter((candidate) => candidate.harness !== harness),
      }))),
      Effect.onError(() => restoreFiles(snapshots).pipe(Effect.ignore)),
    )
  }).pipe(Effect.mapError((error) => error instanceof HarnessConnectionError
    ? error
    : failure("disconnect", String(error), harness))))

  const installSkill: HarnessConnection["installSkill"] = (harness) => {
    const target = registry.get(harness).skillInstallationTarget
    return provide(
      writeFileAtomic(paths.skillInstallations[target].skillFile, skillContents).pipe(
        Effect.mapError((error) => failure("skill", String(error), harness)),
      ),
    )
  }

  return {
    list,
    connect: (harness, options) => withMutationLock(connect(harness, options)),
    sync: (harness) => withMutationLock(sync(harness)),
    disconnect: (harness) => withMutationLock(disconnect(harness)),
    installSkill,
    installStartup,
  } satisfies HarnessConnection
})

export const makeHarnessConnection = makeHarnessConnectionService()
