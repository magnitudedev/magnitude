import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { HarnessIdSchema, type HarnessId, type HarnessLaunchPlan } from "@magnitudedev/client-common"
import {
  MAGNITUDE_ANTHROPIC_BASE_URL,
  MAGNITUDE_INFERENCE_BASE_URL,
  type ProviderModelId,
} from "@magnitudedev/sdk"
import { Effect, Option } from "effect"
import { applyEdits, modify, parse, type ParseError } from "jsonc-parser"
import { isDeepStrictEqual } from "node:util"
import { parseDocument } from "yaml"
import { writeFileAtomic } from "../utils/atomic-file"
import type { HarnessConnectionSpec, HarnessConnector, HarnessInstallation, HarnessModel } from "./contract"

export const OPENAI_BASE_URL = new URL("v1", MAGNITUDE_INFERENCE_BASE_URL).href.replace(/\/$/, "")
export const ANTHROPIC_BASE_URL = MAGNITUDE_ANTHROPIC_BASE_URL.replace(/\/$/, "")
export const LOCAL_TOKEN = "magnitude-local"
export const CHAT_COMPLETIONS_API = "openai-completions" as const
export const OPENAI_COMPATIBLE_PACKAGE = "@ai-sdk/openai-compatible" as const
const LOCAL_ANTHROPIC_MODEL_PREFIX = "anthropic-local/"

export const anthropicLocalModelId = (modelId: string): string =>
  `${LOCAL_ANTHROPIC_MODEL_PREFIX}${modelId}`

export const readOr = (file: string, fallback: string) => FileSystem.FileSystem.pipe(
  Effect.flatMap((fs) => fs.readFileString(file)),
  Effect.catchTag("SystemError", (error) => error.reason === "NotFound"
    ? Effect.succeed(fallback)
    : Effect.fail(error)),
)

export const updateJsonc = (
  source: string,
  changes: ReadonlyArray<readonly [ReadonlyArray<string>, unknown]>,
): string => {
  const errors: ParseError[] = []
  const value = parse(source, errors, { allowTrailingComma: true, disallowComments: false })
  if (errors.length > 0 || value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("configuration is not a JSON object")
  }
  return changes.reduce((current, [segments, next]) => applyEdits(
    current,
    modify(current, [...segments], next, {
      formattingOptions: { insertSpaces: true, tabSize: 2, eol: "\n" },
    }),
  ), source)
}

export const jsonObject = (source: string): Record<string, unknown> => {
  const errors: ParseError[] = []
  const value = parse(source, errors, { allowTrailingComma: true, disallowComments: false })
  if (errors.length > 0 || value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("configuration is not a JSON object")
  }
  return value as Record<string, unknown>
}

export const valueAt = (value: unknown, segments: ReadonlyArray<string>): unknown => segments.reduce<unknown>(
  (current, segment) => current !== null && typeof current === "object"
    ? (current as Record<string, unknown>)[segment]
    : undefined,
  value,
)

export type OwnedValue = readonly [ReadonlyArray<string>, unknown]

export const removeOwnedJsonc = (source: string, owned: ReadonlyArray<OwnedValue>): string => {
  const value = jsonObject(source)
  return owned.reduce((current, [segments, expected]) => {
    if (!isDeepStrictEqual(valueAt(value, segments), expected)) return current
    return applyEdits(current, modify(current, [...segments], undefined, {
      formattingOptions: { insertSpaces: true, tabSize: 2, eol: "\n" },
    }))
  }, source)
}

export const removeJsoncPaths = (
  source: string,
  paths: ReadonlyArray<ReadonlyArray<string>>,
): string => paths.reduce((current, segments) => applyEdits(current, modify(
  current,
  [...segments],
  undefined,
  { formattingOptions: { insertSpaces: true, tabSize: 2, eol: "\n" } },
)), source)

export const updateYaml = (
  source: string,
  changes: ReadonlyArray<readonly [ReadonlyArray<string>, unknown]>,
): string => {
  const document = parseDocument(source.trim() === "" ? "{}\n" : source)
  if (document.errors.length > 0) throw new Error("configuration is not valid YAML")
  for (const [segments, value] of changes) document.setIn([...segments], value)
  return String(document)
}

const deleteYamlPath = (document: ReturnType<typeof parseDocument>, segments: ReadonlyArray<string>): void => {
  document.deleteIn([...segments])
  for (let length = segments.length - 1; length > 0; length -= 1) {
    const parent = segments.slice(0, length)
    const value = valueAt(document.toJS(), parent)
    if (value === null || typeof value !== "object" || Array.isArray(value) || Object.keys(value).length > 0) break
    document.deleteIn([...parent])
  }
}

export const removeYamlPaths = (
  source: string,
  paths: ReadonlyArray<ReadonlyArray<string>>,
): string => {
  const document = parseDocument(source.trim() === "" ? "{}\n" : source)
  if (document.errors.length > 0) throw new Error("configuration is not valid YAML")
  for (const segments of paths) deleteYamlPath(document, segments)
  return String(document)
}

export const writeIfChanged = (file: string, source: string, next: string) => next === source
  ? Effect.void
  : writeFileAtomic(file, next)

export const launchPlan = (
  connector: { readonly id: Parameters<typeof HarnessIdSchema.make>[0] },
  installation: HarnessInstallation,
  modelId: ProviderModelId,
  args: ReadonlyArray<string>,
  environment: Readonly<Record<string, string>> = {},
): HarnessLaunchPlan => ({
  harness: HarnessIdSchema.make(connector.id),
  executable: installation.executable,
  args,
  environment,
  modelId,
})

interface ConnectorDefinition extends Omit<HarnessConnector, "id" | "detect"> {
  readonly id: Parameters<typeof HarnessIdSchema.make>[0]
  readonly executable: string
}

export const defineConnector = (definition: ConnectorDefinition): HarnessConnector => {
  const id = HarnessIdSchema.make(definition.id)
  return {
    id,
    name: definition.name,
    ...(definition.recommended === undefined ? {} : { recommended: definition.recommended }),
    ...(definition.note === undefined ? {} : { note: definition.note }),
    ...(definition.requiresStartup === undefined ? {} : { requiresStartup: definition.requiresStartup }),
    skillInstallationTarget: definition.skillInstallationTarget,
    configurationFiles: definition.configurationFiles,
    connect: definition.connect,
    disconnect: definition.disconnect,
    launch: definition.launch,
    detect: (searchPath) => Effect.sync(() => definition.recommended
      ? Option.some({ executable: process.execPath })
      : Option.fromNullable(Bun.which(definition.executable, { PATH: searchPath })).pipe(
          Option.map((executable) => ({ executable })),
        )),
  }
}

export const currentModel = (spec: HarnessConnectionSpec): Option.Option<ProviderModelId> => spec.setCurrent

export const modelEntries = (models: ReadonlyArray<HarnessModel>): ReadonlyArray<HarnessModel> => models
