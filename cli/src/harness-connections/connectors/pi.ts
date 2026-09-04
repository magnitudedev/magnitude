import * as Command from "@effect/platform/Command"
import { Data, Effect, Option, Schema } from "effect"
import { dirname, resolve } from "node:path"
import type { HarnessConnectionPaths } from "../paths"
import {
  CHAT_COMPLETIONS_API,
  LOCAL_TOKEN,
  OPENAI_BASE_URL,
  defineConnector,
  jsonObject,
  launchPlan,
  readOr,
  qualifiedModelSelection,
  removeJsoncPaths,
  splitQualifiedModelSelection,
  updateJsonc,
  valueAt,
  writeIfChanged,
} from "../shared"
import type {
  HarnessCompanionPackage,
  HarnessConnectionSpec,
} from "../contract"
import { modelInput, modelMaxTokens, zeroCost } from "../model-fields"
import { hasReasoning, projectReasoningControls } from "../reasoning"

const PI_THINKING_SURFACE = {
  controls: ["off", "minimal", "low", "medium", "high", "xhigh", "max"],
  off: "off",
  soleEnabled: "high",
  aliases: { medium: "adaptive" },
} as const

export const PI_COMPANION_PACKAGE_NAME = "Magnitude for Pi"
export const PI_COMPANION_PACKAGE_IDENTITY = "@magnitudedev/pi"
export const PI_COMPANION_PACKAGE_SOURCE = "npm:@magnitudedev/pi@0.0.1"
export const PI_COMPANION_EXTENSION_PATH = "extensions/magnitude.ts"

const packageSource = (entry: unknown): string | undefined => {
  if (typeof entry === "string") return entry
  if (entry === null || typeof entry !== "object" || Array.isArray(entry)) return undefined
  const source = (entry as Record<string, unknown>).source
  return typeof source === "string" ? source : undefined
}

const npmPackageIdentity = (source: string): string | undefined => {
  if (!source.startsWith("npm:")) return undefined
  const specifier = source.slice("npm:".length)
  if (!specifier.startsWith("@")) return specifier.split("@")[0]
  const slash = specifier.indexOf("/")
  if (slash < 0) return undefined
  const version = specifier.indexOf("@", slash)
  return version < 0 ? specifier : specifier.slice(0, version)
}

const packageSourcesEqual = (left: string, right: string, settingsPath: string): boolean => {
  if (left.startsWith("npm:") || right.startsWith("npm:")) return left === right
  const settingsDirectory = dirname(settingsPath)
  return resolve(settingsDirectory, left) === resolve(settingsDirectory, right)
}

const packageEntry = (
  settings: Record<string, unknown>,
  sources: ReadonlyArray<string>,
  settingsPath: string,
) => {
  const packages = Array.isArray(settings.packages) ? settings.packages : []
  const index = packages.findIndex((entry) => {
    const source = packageSource(entry)
    if (source === undefined) return false
    return sources.some((candidate) => packageSourcesEqual(source, candidate, settingsPath)
      || (candidate.startsWith("npm:")
        && npmPackageIdentity(candidate) === PI_COMPANION_PACKAGE_IDENTITY
        && npmPackageIdentity(source) === PI_COMPANION_PACKAGE_IDENTITY))
  })
  return index < 0 ? undefined : { index, entry: packages[index] }
}

const globMatches = (pattern: string, path: string): boolean => {
  try {
    return new Bun.Glob(pattern.replaceAll("\\", "/").replace(/^\.\//, "")).match(path)
  } catch {
    return false
  }
}

const exactFilterTarget = (filter: string): string => filter
  .slice(filter.startsWith("!") || filter.startsWith("+") || filter.startsWith("-") ? 1 : 0)
  .replaceAll("\\", "/")
  .replace(/^\.\//, "")

class PiPackageCommandError extends Data.TaggedError("PiPackageCommandError")<{
  readonly command: "install" | "remove"
  readonly exitCode: number
}> {
  override get message(): string {
    return `Pi package ${this.command} exited with status ${this.exitCode}`
  }
}

export const piPackageExtensionEnabled = (entry: unknown): boolean => {
  if (typeof entry === "string") return true
  if (entry === null || typeof entry !== "object" || Array.isArray(entry)) return false
  const extensions = (entry as Record<string, unknown>).extensions
  if (extensions === undefined) return true
  if (!Array.isArray(extensions) || !extensions.every((value) => typeof value === "string")) return false
  const filters = extensions as string[]
  const forcedInclude = filters.some((filter) => filter.startsWith("+")
    && exactFilterTarget(filter) === PI_COMPANION_EXTENSION_PATH)
  const forcedExclude = filters.some((filter) => filter.startsWith("-")
    && exactFilterTarget(filter) === PI_COMPANION_EXTENSION_PATH)
  if (forcedExclude) return false
  if (forcedInclude) return true
  const positives = filters.filter((value) => !value.startsWith("!") && !value.startsWith("+") && !value.startsWith("-"))
  let included = positives.length === 0 || positives.some((value) => globMatches(value, PI_COMPANION_EXTENSION_PATH))
  if (filters.some((value) => value.startsWith("!") && globMatches(exactFilterTarget(value), PI_COMPANION_EXTENSION_PATH))) {
    included = false
  }
  return included
}

const enablePackageEntry = (entry: unknown): unknown => {
  if (typeof entry === "string") return entry
  if (entry === null || typeof entry !== "object" || Array.isArray(entry)) return entry
  const object = entry as Record<string, unknown>
  const filters = Array.isArray(object.extensions)
      && object.extensions.every((filter) => typeof filter === "string")
    ? object.extensions
    : []
  const forceInclude = `+${PI_COMPANION_EXTENSION_PATH}`
  return {
    ...object,
    extensions: [
      ...filters.filter((filter) => !((filter.startsWith("+") || filter.startsWith("-"))
          && exactFilterTarget(filter) === PI_COMPANION_EXTENSION_PATH)),
      forceInclude,
    ],
  }
}

const runPiPackageCommand = (executable: string, command: "install" | "remove", source: string) =>
  Command.make(executable, command, source).pipe(
    Command.exitCode,
    Effect.timeout("2 minutes"),
    Effect.flatMap((exitCode) => Number(exitCode) === 0
      ? Effect.void
      : Effect.fail(new PiPackageCommandError({ command, exitCode: Number(exitCode) }))),
  )

const JsonDocumentSchema = Schema.parseJson(Schema.Unknown)
const parseJsonDocument = Schema.decodeUnknownSync(JsonDocumentSchema)
const stringifyJsonDocument = Schema.encodeSync(JsonDocumentSchema)

export const makePiCompanion = (
  paths: HarnessConnectionPaths,
  desiredSource = PI_COMPANION_PACKAGE_SOURCE,
): HarnessCompanionPackage => ({
  description: {
    name: PI_COMPANION_PACKAGE_NAME,
    source: desiredSource,
    securityNotice: "Pi extensions execute with your user permissions.",
  },
  activation: "reload-or-restart",
  reconcile: ({ installation, previous }) => Effect.gen(function* () {
    const settingsSource = yield* readOr(paths.piSettings, "{}\n")
    const settings = jsonObject(settingsSource)
    const current = packageEntry(settings, [
      desiredSource,
      ...Option.match(previous, { onNone: () => [], onSome: ({ source }) => [source] }),
    ], paths.piSettings)
    if (Option.isSome(previous)) {
      if (current === undefined) {
        const source = previous.value.ownership === "magnitude"
          ? desiredSource
          : Option.getOrElse(Option.flatMap(
              previous.value.previousEntryJson,
              (json) => Option.fromNullable(packageSource(parseJsonDocument(json))),
            ), () => previous.value.source)
        yield* runPiPackageCommand(
          installation.executable,
          "install",
          source,
        )
        return {
          state: { ...previous.value, source },
          status: "installed" as const,
          rollback: runPiPackageCommand(
            installation.executable,
            "remove",
            source,
          ),
        }
      }
      const currentSource = packageSource(current.entry) ?? previous.value.source
      if (previous.value.ownership === "magnitude"
          && !packageSourcesEqual(currentSource, desiredSource, paths.piSettings)) {
        yield* runPiPackageCommand(installation.executable, "remove", previous.value.source)
        yield* runPiPackageCommand(installation.executable, "install", desiredSource).pipe(
          Effect.onError(() => runPiPackageCommand(
            installation.executable,
            "install",
            previous.value.source,
          ).pipe(Effect.ignore)),
        )
        return {
          state: { ...previous.value, source: desiredSource },
          status: "installed" as const,
          rollback: runPiPackageCommand(installation.executable, "remove", desiredSource).pipe(
            Effect.ignore,
            Effect.zipRight(runPiPackageCommand(installation.executable, "install", previous.value.source)),
          ),
        }
      }
      if (!piPackageExtensionEnabled(current.entry)) {
        yield* writeIfChanged(paths.piSettings, settingsSource, updateJsonc(settingsSource, [[
          ["packages", current.index], enablePackageEntry(current.entry),
        ]]))
        return { state: previous.value, status: "enabled" as const, rollback: Effect.void }
      }
      return { state: previous.value, status: "already-installed" as const, rollback: Effect.void }
    }
    if (current === undefined) {
      yield* runPiPackageCommand(installation.executable, "install", desiredSource)
      return {
        state: {
          identity: PI_COMPANION_PACKAGE_IDENTITY,
          source: desiredSource,
          ownership: "magnitude" as const,
          previousEntryJson: Option.none<string>(),
        },
        status: "installed" as const,
        rollback: runPiPackageCommand(
          installation.executable,
          "remove",
          desiredSource,
        ),
      }
    }
    const currentSource = packageSource(current.entry) ?? desiredSource
    const source = packageSourcesEqual(currentSource, desiredSource, paths.piSettings)
      ? desiredSource
      : currentSource
    const enabled = piPackageExtensionEnabled(current.entry)
    if (!enabled) {
      yield* writeIfChanged(paths.piSettings, settingsSource, updateJsonc(settingsSource, [[
        ["packages", current.index], enablePackageEntry(current.entry),
      ]]))
    }
    return {
      state: {
        identity: PI_COMPANION_PACKAGE_IDENTITY,
        source,
        ownership: "pre-existing" as const,
        previousEntryJson: Option.some(stringifyJsonDocument(current.entry)),
      },
      status: enabled ? "already-installed" as const : "enabled" as const,
      rollback: Effect.void,
    }
  }),
  disconnect: ({ installation, state }) => Effect.gen(function* () {
    if (state.ownership === "magnitude") {
      yield* runPiPackageCommand(
        installation.executable,
        "remove",
        state.source,
      )
      return {
        rollback: runPiPackageCommand(
          installation.executable,
          "install",
          state.source,
        ),
      }
    }
    if (Option.isNone(state.previousEntryJson)) return { rollback: Effect.void }
    const settingsSource = yield* readOr(paths.piSettings, "{}\n")
    const settings = jsonObject(settingsSource)
    const current = packageEntry(settings, [state.source], paths.piSettings)
    const previousEntry = parseJsonDocument(state.previousEntryJson.value)
    const packages = Array.isArray(settings.packages) ? settings.packages : []
    const next = current === undefined
      ? [...packages, previousEntry]
      : packages.map((entry, index) => index === current.index ? previousEntry : entry)
    yield* writeIfChanged(paths.piSettings, settingsSource, updateJsonc(settingsSource, [[
      ["packages"], next,
    ]]))
    return { rollback: Effect.void }
  }),
})

export const piModels = (models: HarnessConnectionSpec["models"]) => models.map((model) => {
  const reasoning = hasReasoning(model)
  return {
    id: model.id,
    name: model.name,
    reasoning,
    thinkingLevelMap: projectReasoningControls(model, PI_THINKING_SURFACE).map,
    input: modelInput(model),
    cost: zeroCost(),
    contextWindow: model.contextWindow,
    maxTokens: modelMaxTokens(model),
    compat: { supportsReasoningEffort: reasoning },
  }
})

export const piProviderConfig = (models: HarnessConnectionSpec["models"]) => ({
  baseUrl: OPENAI_BASE_URL,
  api: CHAT_COMPLETIONS_API,
  apiKey: LOCAL_TOKEN,
  models: piModels(models),
})

export interface PiConnectorOptions {
  readonly companion?: HarnessCompanionPackage
  readonly packageSource?: string
}

export const makePiConnector = (
  paths: HarnessConnectionPaths,
  options: PiConnectorOptions = {},
) => defineConnector({
  id: "pi",
  name: "Pi",
  executable: "pi",
  skillInstallationTarget: "shared-agents",
  skillRequired: true,
  companion: options.companion ?? makePiCompanion(paths, options.packageSource),
  configurationFiles: [paths.piModels, paths.piSettings],
  connect: (spec) => Effect.gen(function* () {
    const source = yield* readOr(paths.piModels, "{}\n")
    yield* writeIfChanged(paths.piModels, source, updateJsonc(source, [[
      ["providers", "magnitude"], piProviderConfig(spec.models),
    ]]))
    if (Option.isNone(spec.model)) return Option.none()
    const settingsSource = yield* readOr(paths.piSettings, "{}\n")
    const settings = jsonObject(settingsSource)
    const restore = {
      model: qualifiedModelSelection(
        valueAt(settings, ["defaultProvider"]),
        valueAt(settings, ["defaultModel"]),
      ),
    }
    yield* writeIfChanged(paths.piSettings, settingsSource, updateJsonc(settingsSource, [
      [["defaultProvider"], "magnitude"],
      [["defaultModel"], spec.model.value],
    ]))
    return Option.some(restore)
  }),
  disconnect: (spec) => Effect.gen(function* () {
    const settingsSource = yield* readOr(paths.piSettings, "{}\n")
    const settings = jsonObject(settingsSource)
    if (valueAt(settings, ["defaultProvider"]) === "magnitude" && Option.isSome(spec.restore)) {
      const previous = Option.flatMap(spec.restore.value.model, (selection) =>
        Option.fromNullable(splitQualifiedModelSelection(selection)))
      yield* writeIfChanged(paths.piSettings, settingsSource, updateJsonc(settingsSource, [
        [["defaultProvider"], Option.match(previous, { onNone: () => undefined, onSome: ({ provider }) => provider })],
        [["defaultModel"], Option.match(previous, { onNone: () => undefined, onSome: ({ model }) => model })],
      ]))
    }
    const source = yield* readOr(paths.piModels, "{}\n")
    yield* writeIfChanged(paths.piModels, source, removeJsoncPaths(source, [["providers", "magnitude"]]))
  }),
  launch(modelId, installation) {
    return launchPlan(this, installation, modelId, ["--model", `magnitude/${modelId}`])
  },
})
