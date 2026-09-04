import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { HarnessIdSchema, type HarnessId, type HarnessLaunchPlan } from "@magnitudedev/client-common"
import {
  MAGNITUDE_CLAUDE_CODE_PROXY_BASE_URL,
  MAGNITUDE_CODEX_PROXY_BASE_URL,
  MAGNITUDE_INFERENCE_BASE_URL,
  type ProviderModelId,
} from "@magnitudedev/sdk"
import { Effect, Option } from "effect"
import { applyEdits, modify, parse, type ParseError } from "jsonc-parser"
import { isDeepStrictEqual } from "node:util"
import { parseDocument } from "yaml"
import { writeFileAtomic } from "./configuration-file"
import type { HarnessConnector, HarnessInstallation } from "./contract"

export const OPENAI_BASE_URL = new URL("v1", MAGNITUDE_INFERENCE_BASE_URL).href.replace(/\/$/, "")
export const CODEX_PROXY_BASE_URL = MAGNITUDE_CODEX_PROXY_BASE_URL.replace(/\/$/, "")
export const ANTHROPIC_BASE_URL = MAGNITUDE_CLAUDE_CODE_PROXY_BASE_URL.replace(/\/$/, "")
export const LOCAL_TOKEN = "magnitude-local"
export const CHAT_COMPLETIONS_API = "openai-completions" as const
export const OPENAI_COMPATIBLE_PACKAGE = "@ai-sdk/openai-compatible" as const
const LOCAL_ANTHROPIC_MODEL_PREFIX = "anthropic-local/"
const LOCAL_CODEX_MODEL_PREFIX = "magnitude-local/"

export const anthropicLocalModelId = (modelId: string): string =>
  `${LOCAL_ANTHROPIC_MODEL_PREFIX}${modelId}`

export const isAnthropicLocalModelId = (modelId: unknown): modelId is string =>
  typeof modelId === "string" && modelId.startsWith(LOCAL_ANTHROPIC_MODEL_PREFIX)

export const codexLocalModelId = (modelId: string): string =>
  `${LOCAL_CODEX_MODEL_PREFIX}${modelId}`

export const isCodexLocalModelId = (modelId: unknown): modelId is string =>
  typeof modelId === "string" && modelId.startsWith(LOCAL_CODEX_MODEL_PREFIX)

/** Normalize a harness selection whose provider and model are stored separately. */
export const qualifiedModelSelection = (provider: unknown, model: unknown): Option.Option<string> =>
  typeof provider === "string" && provider.length > 0 && typeof model === "string" && model.length > 0
    ? Option.some(`${provider}/${model}`)
    : Option.none()

export const splitQualifiedModelSelection = (
  selection: string,
): { readonly provider: string; readonly model: string } | undefined => {
  const separator = selection.indexOf("/")
  return separator <= 0 || separator === selection.length - 1
    ? undefined
    : { provider: selection.slice(0, separator), model: selection.slice(separator + 1) }
}

export const tomlTopLevelValue = (source: string, key: string): unknown => {
  const value = Bun.TOML.parse(source.trim() === "" ? "" : source)
  return value !== null && typeof value === "object" ? (value as Record<string, unknown>)[key] : undefined
}

const tomlScalar = (value: string): string => JSON.stringify(value)

/** Update string-valued top-level TOML keys while preserving unrelated text and tables. */
export const updateTomlTopLevel = (
  source: string,
  changes: ReadonlyArray<readonly [string, string | undefined]>,
): string => {
  const lines = source.split("\n")
  const additions: string[] = []
  for (const [key, value] of changes) {
    const matcher = new RegExp(`^\\s*${key}\\s*=`)
    const firstTable = lines.findIndex((line) => /^\s*\[/.test(line))
    const boundary = firstTable === -1 ? lines.length : firstTable
    const index = lines.slice(0, boundary).findIndex((line) => matcher.test(line))
    if (index >= 0) {
      if (value === undefined) lines.splice(index, 1)
      else lines[index] = `${key} = ${tomlScalar(value)}`
    } else if (value !== undefined) {
      additions.push(`${key} = ${tomlScalar(value)}`)
    }
  }
  if (additions.length > 0) {
    const insertion = lines.findIndex((line) => /^\s*\[/.test(line))
    lines.splice(insertion === -1 ? Math.max(0, lines.length - 1) : insertion, 0, ...additions)
  }
  return lines.join("\n")
}

export const removeTomlTable = (source: string, table: string): string => {
  const lines = source.split("\n")
  const header = `[${table}]`
  const start = lines.findIndex((line) => line.trim() === header)
  if (start < 0) return source
  let end = start + 1
  while (end < lines.length && !/^\s*\[/.test(lines[end]!)) end += 1
  lines.splice(start, end - start)
  return lines.join("\n")
}

/** Find a [[table]] array-of-tables block where `name = "X"` matches, returning line boundaries. */
const findTomlArrayBlock = (
  lines: string[],
  table: string,
  name: string,
): { start: number; end: number } | undefined => {
  const arrayHeader = new RegExp(`^\\s*\\[\\[${table}\\]\\]\\s*$`)
  for (let i = 0; i < lines.length; i++) {
    if (!arrayHeader.test(lines[i]!)) continue
    // Scan forward until the next section header to check for name match
    let j = i + 1
    let nameMatch = false
    while (j < lines.length && !/^\s*\[/.test(lines[j]!)) {
      const m = lines[j]!.match(/^\s*name\s*=\s*"([^"]*)"/)
      if (m?.[1] === name) nameMatch = true
      j++
    }
    if (nameMatch) return { start: i, end: j }
  }
  return undefined
}

/**
 * Replace an existing [[table]] block where name matches with `block`,
 * or append `block` if no matching block exists.
 */
export const replaceOrAppendTomlArrayBlock = (
  source: string,
  table: string,
  name: string,
  block: string,
): string => {
  const lines = source.split("\n")
  const existing = findTomlArrayBlock(lines, table, name)
  const blockLines = block.trimEnd().split("\n")
  if (existing !== undefined) {
    lines.splice(existing.start, existing.end - existing.start, ...blockLines)
    const result = lines.join("\n")
    // Preserve trailing newline from source so idempotent calls don't flip it
    return source.endsWith("\n") && !result.endsWith("\n") ? `${result}\n` : result
  }
  const trimmed = source.trimEnd()
  return `${trimmed}${trimmed.length > 0 ? "\n\n" : ""}${block.trimEnd()}\n`
}

/**
 * Remove the [[table]] block where name matches, trimming any preceding blank line.
 */
export const removeTomlArrayBlock = (source: string, table: string, name: string): string => {
  const lines = source.split("\n")
  const existing = findTomlArrayBlock(lines, table, name)
  if (existing === undefined) return source
  let { start } = existing
  const { end } = existing
  // Absorb a preceding blank separator line so the result stays clean
  if (start > 0 && lines[start - 1]!.trim() === "") start--
  lines.splice(start, end - start)
  return lines.join("\n")
}

/** Read a scalar value from a named TOML table, e.g. `[models].default`. */
export const tomlTableScalarValue = (source: string, table: string, key: string): unknown => {
  const parsed = Bun.TOML.parse(source.trim() === "" ? "" : source)
  if (parsed === null || typeof parsed !== "object") return undefined
  const section = (parsed as Record<string, unknown>)[table]
  return section !== null && typeof section === "object"
    ? (section as Record<string, unknown>)[key]
    : undefined
}

/**
 * Set or remove string scalars within a named TOML table section, e.g. `[models]`.
 * If the table does not exist and values are being set, it is appended.
 */
export const updateTomlTableScalar = (
  source: string,
  table: string,
  changes: ReadonlyArray<readonly [string, string | undefined]>,
): string => {
  const tableHeader = `[${table}]`
  const lines = source.split("\n")
  const tableStart = lines.findIndex((line) => line.trim() === tableHeader)
  if (tableStart < 0) {
    const additions = changes.filter(([, v]) => v !== undefined)
    if (additions.length === 0) return source
    const block = `\n${tableHeader}\n${additions.map(([k, v]) => `${k} = ${JSON.stringify(v)}`).join("\n")}\n`
    return source.trimEnd() + block
  }
  let tableEnd = tableStart + 1
  while (tableEnd < lines.length && !/^\s*\[/.test(lines[tableEnd]!)) tableEnd++
  const sectionLines = lines.slice(tableStart + 1, tableEnd)
  const additions: string[] = []
  for (const [key, value] of changes) {
    const matcher = new RegExp(`^\\s*${key}\\s*=`)
    const idx = sectionLines.findIndex((line) => matcher.test(line))
    if (idx >= 0) {
      if (value === undefined) sectionLines.splice(idx, 1)
      else sectionLines[idx] = `${key} = ${JSON.stringify(value)}`
    } else if (value !== undefined) {
      additions.push(`${key} = ${JSON.stringify(value)}`)
    }
  }
  if (additions.length > 0) sectionLines.push(...additions)
  lines.splice(tableStart + 1, tableEnd - tableStart - 1, ...sectionLines)
  return lines.join("\n")
}

export const readOr = (file: string, fallback: string) => FileSystem.FileSystem.pipe(
  Effect.flatMap((fs) => fs.readFileString(file)),
  Effect.catchTag("SystemError", (error) => error.reason === "NotFound"
    ? Effect.succeed(fallback)
    : Effect.fail(error)),
)

export const updateJsonc = (
  source: string,
  changes: ReadonlyArray<readonly [ReadonlyArray<string | number>, unknown]>,
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

export const removeOwnedYaml = (
  source: string,
  owned: ReadonlyArray<OwnedValue>,
): string => {
  const document = parseDocument(source.trim() === "" ? "{}\n" : source)
  if (document.errors.length > 0) throw new Error("configuration is not valid YAML")
  for (const [segments, expected] of owned) {
    if (isDeepStrictEqual(valueAt(document.toJS(), segments), expected)) {
      deleteYamlPath(document, segments)
    }
  }
  return String(document)
}

export const writeIfChanged = (file: string, source: string, next: string) => next === source
  ? Effect.void
  : writeFileAtomic(file, next)

export const launchPlan = (
  connector: {
    readonly id: Parameters<typeof HarnessIdSchema.make>[0]
    readonly executable: string
  },
  installation: HarnessInstallation,
  modelId: ProviderModelId,
  args: ReadonlyArray<string>,
  environment: Readonly<Record<string, string>> = {},
): HarnessLaunchPlan => ({
  harness: HarnessIdSchema.make(connector.id),
  command: connector.executable,
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
    executable: definition.executable,
    ...(definition.recommended === undefined ? {} : { recommended: definition.recommended }),
    ...(definition.note === undefined ? {} : { note: definition.note }),
    ...(definition.requiresStartup === undefined ? {} : { requiresStartup: definition.requiresStartup }),
    ...(definition.companion === undefined ? {} : { companion: definition.companion }),
    ...(definition.skillRequired === true ? { skillRequired: true } : {}),
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
