import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import * as Path from "@effect/platform/Path"
import {
  HarnessConnectionError,
  HarnessIdSchema,
  type HarnessConnection,
  type HarnessDestination,
  type HarnessId,
} from "@magnitudedev/client-common"
import {
  makeInferenceClient,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  type InferenceModel,
  type ProviderModelId,
} from "@magnitudedev/sdk"
import { makeStateDocument } from "@magnitudedev/storage"
import { Effect, Option, Schema } from "effect"
import { delimiter } from "node:path"
import { installServiceOnStartup } from "../server/service"
import { writeFileAtomic } from "../utils/atomic-file"
import {
  HarnessModelSchema,
  HarnessCompanionStateSchema,
  type HarnessConnectionSpec,
  type HarnessConnector,
  type HarnessInstallation,
  type HarnessModel,
  type HarnessCompanionState,
  type HarnessRestore,
  HarnessRestoreSchema,
} from "./contract"
import { harnessConnectionPaths, type HarnessConnectionPaths } from "./paths"
import { makeHarnessConnectorRegistry, type HarnessConnectorRegistry } from "./registry"
import { OPENAI_BASE_URL, readOr } from "./shared"
import skillContents from "./magnitude-skill.md" with { type: "text" }

export {
  ANTHROPIC_BASE_URL,
  CODEX_PROXY_BASE_URL,
  OPENAI_BASE_URL,
  anthropicLocalModelId,
  codexLocalModelId,
  removeOwnedJsonc,
  updateJsonc,
  updateYaml,
} from "./shared"
export { clineModelCatalog, clineModelRegistryEntry, clineProviderSettings } from "./connectors/cline"
export { CLAUDE_GATEWAY_DISCOVERY } from "./connectors/claude-code"
export { codexModelCatalog } from "./connectors/codex"
export { hermesProviderConfig, hermesReasoningOverrides } from "./connectors/hermes"
export { ohMyPiProviderConfig } from "./connectors/oh-my-pi"
export { openClawAgentConfig, openClawProviderConfig } from "./connectors/openclaw"
export { openCodeProviderConfig } from "./connectors/opencode"
export {
  PI_COMPANION_EXTENSION_PATH,
  PI_COMPANION_PACKAGE_IDENTITY,
  PI_COMPANION_PACKAGE_SOURCE,
  makePiCompanion,
  piPackageExtensionEnabled,
  piProviderConfig,
} from "./connectors/pi"
export { makeHarnessConnectorRegistry } from "./registry"
export { harnessConnectionPaths, type HarnessConnectionPaths } from "./paths"
export type { HarnessConnectionSpec, HarnessConnector, HarnessInstallation, HarnessModel } from "./contract"

const ManifestEntrySchema = Schema.Struct({
  harness: HarnessIdSchema,
  models: Schema.optionalWith(Schema.Array(HarnessModelSchema), { default: () => [] }),
  restore: Schema.optionalWith(HarnessRestoreSchema, { as: "Option", exact: true }),
  companion: Schema.optionalWith(HarnessCompanionStateSchema, { as: "Option", exact: true }),
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
  readonly registry?: HarnessConnectorRegistry
  readonly installStartup?: Effect.Effect<
    void,
    unknown,
    FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor
  >
}

const upsertManifest = (
  manifest: Manifest,
  harness: HarnessId,
  models: ReadonlyArray<HarnessModel>,
  restore: Option.Option<HarnessRestore>,
  companion: Option.Option<HarnessCompanionState>,
  now: () => Date,
) => ({
  connections: [
    ...manifest.connections.filter((entry) => entry.harness !== harness),
    { harness, models: [...models], restore, companion, updatedAt: now().toISOString() },
  ],
}) satisfies Manifest

const entrySpec = (entry: ManifestEntry) => ({
  models: entry.models,
  restore: entry.restore,
})

const toHarnessModel = (model: InferenceModel): HarnessModel => {
  const reasoning = Option.match(Option.filter(model.reasoning, (value) => value !== null), {
    onNone: () => ({ supported: false as const, efforts: [] as const }),
    onSome: (value) => ({
      supported: true as const,
      efforts: value.supported_efforts.map((effort) => Schema.decodeSync(ReasoningEffortSchema)(effort)),
      defaultEffort: Schema.decodeSync(ReasoningEffortSchema)(value.default_effort),
    }),
  })
  return Schema.decodeSync(HarnessModelSchema)({
    id: model.id,
    name: model.name,
    description: model.description,
    contextWindow: model.context_length,
    maxOutputTokens: model.top_provider.max_completion_tokens,
    capabilities: {
      vision: model.architecture.input_modalities.includes("image"),
      tools: model.supported_parameters.includes("tools"),
      structuredOutput: model.supported_parameters.includes("structured_outputs"),
      reasoning,
    },
  })
}

const discoverMagnitudeModels = Effect.gen(function* () {
  const client = yield* makeInferenceClient()
  const response = yield* client.listModels()
  return response.data.map(toHarnessModel)
})

const uniqueModels = (models: ReadonlyArray<HarnessModel>): ReadonlyArray<HarnessModel> => models.filter(
  (model, index) => models.findIndex((candidate) => candidate.id === model.id) === index,
)

const containsModel = (models: ReadonlyArray<HarnessModel>, modelId: ProviderModelId): boolean =>
  models.some(({ id }) => id === modelId)

export const makeHarnessConnectionService = (options: HarnessConnectionOptions = {}) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const runtime = yield* Effect.runtime<
    FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor | HttpClient.HttpClient
  >()
  const paths = options.paths ?? harnessConnectionPaths()
  const localPiPackageSource = process.env.MAGNITUDE_PI_PACKAGE_SOURCE?.trim()
  const registry = options.registry ?? makeHarnessConnectorRegistry(paths, {
    ...(localPiPackageSource === undefined || localPiPackageSource.length === 0
      ? {}
      : { piCompanionSource: localPiPackageSource }),
  })
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

  const connectorOperation = <A>(
    operation: HarnessConnectionError["operation"],
    connector: HarnessConnector,
    effect: Effect.Effect<
      A,
      unknown,
      FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor
    >,
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

  const list: HarnessConnection["list"] = provide(Effect.gen(function* () {
    const manifest = yield* readManifest
    const connected = new Set(manifest.connections.map(({ harness }) => harness))
    return yield* Effect.forEach(registry.ordered, (connector) => installed(connector).pipe(
      Effect.map((installation): HarnessDestination => ({
        id: connector.id,
        name: connector.name,
        availability: Option.isSome(installation) ? "Installed" : "Not installed",
        selectable: connector.recommended || Option.isSome(installation),
        connected: connected.has(connector.id),
        ...(connector.note === undefined ? {} : { note: connector.note }),
        ...(connector.companion === undefined ? {} : { companion: connector.companion.description }),
        ...(connector.skillRequired === true ? { skillRequired: true } : {}),
      })),
    ))
  })).pipe(
    Effect.map((rows) => [...rows.filter((row) => row.selectable), ...rows.filter((row) => !row.selectable)]),
    Effect.mapError((error) => failure("list", String(error))),
  )

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
    if (Option.isSome(connectOptions.model) && !containsModel(models, connectOptions.model.value)) {
      return yield* failure("connect", `Magnitude model is not installed: ${connectOptions.model.value}`, harness)
    }
    const manifest = yield* readManifest
    const existing = manifest.connections.find((entry) => entry.harness === harness)
    const previousModels = existing?.models
    const spec: HarnessConnectionSpec = {
      models,
      model: connectOptions.model,
      installation: installation.value,
      ...(previousModels === undefined ? {} : { previousModels }),
    }
    const shouldInstallStartup = connectOptions.launchOnStartup === true || connector.requiresStartup === true
    const shouldInstallSkill = connector.skillRequired === true || connectOptions.installSkill === true
    if (connector.recommended) {
      if (shouldInstallStartup) yield* installStartup
      if (shouldInstallSkill) yield* installSkill(harness)
      return {
        companion: Option.none(),
        skillInstalled: shouldInstallSkill,
        startupInstalled: shouldInstallStartup,
      }
    }
    const skillFile = paths.skillInstallations[connector.skillInstallationTarget].skillFile
    const snapshots = yield* snapshotFiles([
      ...connector.configurationFiles,
      ...(shouldInstallSkill ? [skillFile] : []),
    ])
    const companionResult = connector.companion === undefined
      ? Option.none<{
          readonly state: HarnessCompanionState
          readonly status: "installed" | "enabled" | "already-installed"
          readonly rollback: Effect.Effect<
            void,
            unknown,
            FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor
          >
        }>()
      : Option.some(yield* connectorOperation(
          "connect",
          connector,
          connector.companion.reconcile({
            installation: installation.value,
            previous: Option.fromNullable(existing).pipe(Option.flatMap(({ companion }) => companion)),
          }),
        ))
    const rollback = Effect.gen(function* () {
      if (Option.isSome(companionResult)) yield* companionResult.value.rollback.pipe(Effect.ignore)
      yield* restoreFiles(snapshots).pipe(Effect.ignore)
    })
    yield* Effect.gen(function* () {
      if (shouldInstallStartup) yield* installStartup
      if (shouldInstallSkill) yield* installSkill(harness)
      const captured = yield* connectorOperation("connect", connector, connector.connect(spec))
      const restore = existing !== undefined && Option.isSome(existing.restore)
        ? existing.restore
        : captured
      const companion = Option.match(companionResult, {
        onNone: () => Option.none<HarnessCompanionState>(),
        onSome: ({ state }) => Option.some(state),
      })
      yield* updateManifest((manifest) => upsertManifest(
        manifest,
        harness,
        models,
        restore,
        companion,
        now,
      ))
    }).pipe(Effect.onError(() => rollback))
    return {
      companion: Option.flatMap(companionResult, ({ status }) => connector.companion === undefined
        ? Option.none()
        : Option.some({
            ...connector.companion.description,
            status,
            activation: connector.companion.activation,
          })),
      skillInstalled: shouldInstallSkill,
      startupInstalled: shouldInstallStartup,
    }
  }).pipe(Effect.mapError((error) => error instanceof HarnessConnectionError
    ? error
    : failure("connect", String(error), harness))))

  const launch: HarnessConnection["launch"] = (harness, modelId) => provide(Effect.gen(function* () {
    const connector = registry.get(harness)
    const installation = yield* installed(connector)
    if (Option.isNone(installation)) return yield* failure("launch", `${connector.name} is not installed`, harness)
    return connector.launch(modelId, installation.value)
  }).pipe(Effect.mapError((error) => error instanceof HarnessConnectionError
    ? error
    : failure("launch", String(error), harness))))

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
      const installation = connector.id === "codex" || connector.companion !== undefined
        ? yield* installed(connector)
        : Option.some({ executable: connector.executable })
      if (Option.isNone(installation)) return yield* failure(
        "sync",
        `${connector.name} is not installed`,
        connector.id,
      )
      const spec: HarnessConnectionSpec = {
        models,
        model: Option.none(),
        installation: installation.value,
        previousModels: entry.models,
      }
      if (connector.requiresStartup) yield* installStartup
      const requiredSkillFile = connector.skillRequired === true
        ? [paths.skillInstallations[connector.skillInstallationTarget].skillFile]
        : []
      const snapshots = yield* snapshotFiles([...connector.configurationFiles, ...requiredSkillFile])
      const companionResult = connector.companion === undefined
        ? Option.none<{
            readonly state: HarnessCompanionState
            readonly rollback: Effect.Effect<
              void,
              unknown,
              FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor
            >
          }>()
        : Option.some(yield* connectorOperation("sync", connector, connector.companion.reconcile({
            installation: installation.value,
            previous: entry.companion,
          })))
      const rollback = Effect.gen(function* () {
        if (Option.isSome(companionResult)) yield* companionResult.value.rollback.pipe(Effect.ignore)
        yield* restoreFiles(snapshots).pipe(Effect.ignore)
      })
      yield* Effect.gen(function* () {
        if (connector.skillRequired === true) yield* installSkill(entry.harness)
        yield* connectorOperation("sync", connector, connector.connect(spec))
        yield* updateManifest((current) => upsertManifest(
          current,
          entry.harness,
          models,
          entry.restore,
          Option.match(companionResult, {
            onNone: () => Option.none(),
            onSome: ({ state }) => Option.some(state),
          }),
          now,
        ))
      }).pipe(Effect.onError(() => rollback))
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
    const installation = connector.companion === undefined
      ? Option.none<HarnessInstallation>()
      : yield* installed(connector)
    if (connector.companion !== undefined && Option.isNone(installation)) {
      return yield* failure("disconnect", `${connector.name} is not installed`, harness)
    }
    const snapshots = yield* snapshotFiles(connector.configurationFiles)
    const companionResult = connector.companion !== undefined
        && Option.isSome(entry.companion)
        && Option.isSome(installation)
      ? Option.some(yield* connectorOperation("disconnect", connector, connector.companion.disconnect({
            installation: installation.value,
            state: entry.companion.value,
          })))
      : Option.none<{ readonly rollback: Effect.Effect<
          void,
          unknown,
          FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor
        > }>()
    const rollback = Effect.gen(function* () {
      if (Option.isSome(companionResult)) yield* companionResult.value.rollback.pipe(Effect.ignore)
      yield* restoreFiles(snapshots).pipe(Effect.ignore)
    })
    yield* connectorOperation("disconnect", connector, connector.disconnect(entrySpec(entry))).pipe(
      Effect.zipRight(updateManifest((current) => ({
        connections: current.connections.filter((candidate) => candidate.harness !== harness),
      }))),
      Effect.onError(() => rollback),
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
    launch,
    sync: (harness?: HarnessId) => withMutationLock(sync(harness)),
    disconnect: (harness) => withMutationLock(disconnect(harness)),
    installSkill,
    installStartup,
  } satisfies HarnessConnection
})

export const makeHarnessConnection = makeHarnessConnectionService()
