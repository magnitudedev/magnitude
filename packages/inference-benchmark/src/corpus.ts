import * as FileSystem from "@effect/platform/FileSystem"
import { Data, Effect, Schema } from "effect"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"
import { homedir } from "node:os"
import type { ChatMessage, ChatTool } from "@magnitudedev/ai"
import type { ExpectedToolCall, Fixture } from "./domain"
import { digestObject, sha256 } from "./hash"

const Digest = Schema.String.pipe(Schema.pattern(/^[0-9a-f]{64}$/))
const CorpusFileLockSchema = Schema.Struct({
  path: Schema.String.pipe(Schema.minLength(1)),
  sha256: Digest,
})
const CorpusLockSchema = Schema.Struct({
  repository: Schema.String.pipe(Schema.minLength(1)),
  commit: Schema.String.pipe(Schema.minLength(1)),
  dataRoot: Schema.String.pipe(Schema.minLength(1)),
  license: Schema.String.pipe(Schema.minLength(1)),
  files: Schema.Array(CorpusFileLockSchema).pipe(Schema.minItems(1)),
})
const CorpusSelectionSchema = Schema.Struct({
  idPrefixes: Schema.Array(Schema.String).pipe(Schema.minItems(1)),
  maximumRecords: Schema.Int.pipe(Schema.greaterThan(0)),
  exclusions: Schema.Array(Schema.String),
})
const JsonObjectSchema = Schema.Record({ key: Schema.String, value: Schema.Unknown })

export type CorpusFileLock = typeof CorpusFileLockSchema.Type
export type CorpusLock = typeof CorpusLockSchema.Type
export type CorpusSelection = typeof CorpusSelectionSchema.Type

export interface PreparedCorpus {
  readonly root: string
  readonly lock: CorpusLock
  readonly selection: CorpusSelection
  readonly fixtures: readonly Fixture[]
  readonly digest: string
}

export class CorpusError extends Data.TaggedError("CorpusError")<{
  readonly operation: string
  readonly message: string
}> {}

const packageRoot = join(dirname(fileURLToPath(import.meta.url)), "..")
const defaultLockPath = join(packageRoot, "corpus", "bfcl-v4.lock.json")
const defaultSelectionPath = join(packageRoot, "corpus", "bfcl-v4.selection.json")

function cacheRoot(): string {
  const explicit = process.env.MAGNITUDE_BENCHMARK_CACHE
  if (explicit) return explicit
  const xdg = process.env.XDG_CACHE_HOME
  return join(xdg ?? join(homedir(), ".cache"), "magnitude", "inference-benchmark")
}

const readJson = <A, I>(
  path: string,
  schema: Schema.Schema<A, I>,
): Effect.Effect<A, CorpusError, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const text = yield* fs.readFileString(path)
    return yield* Schema.decodeUnknown(Schema.parseJson(schema))(text)
  }).pipe(Effect.mapError((error) => error instanceof CorpusError
    ? error
    : new CorpusError({ operation: "read-json", message: `${path}: ${String(error)}` })))

function rawUrl(lock: CorpusLock, path: string): string {
  const repositoryPath = new URL(lock.repository).pathname.replace(/^\//, "").replace(/\.git$/, "")
  return `https://raw.githubusercontent.com/${repositoryPath}/${lock.commit}/${lock.dataRoot}/${path}`
}

const ensureFile = (
  root: string,
  lock: CorpusLock,
  file: CorpusFileLock,
): Effect.Effect<string, CorpusError, FileSystem.FileSystem> => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const destination = join(root, file.path)
  const existing = yield* fs.readFile(destination).pipe(Effect.option)
  if (existing._tag === "Some" && sha256(existing.value) === file.sha256) {
    return destination
  }

  const response = yield* Effect.tryPromise({
    try: () => fetch(rawUrl(lock, file.path)),
    catch: (error) => new CorpusError({
      operation: "download",
      message: `${file.path}: ${error instanceof Error ? error.message : String(error)}`,
    }),
  })
  if (!response.ok) {
    return yield* new CorpusError({ operation: "download", message: `${file.path} returned HTTP ${response.status}` })
  }
  const bytes = new Uint8Array(yield* Effect.tryPromise({
    try: () => response.arrayBuffer(),
    catch: (error) => new CorpusError({
      operation: "download",
      message: `${file.path}: ${error instanceof Error ? error.message : String(error)}`,
    }),
  }))
  const actual = sha256(bytes)
  if (actual !== file.sha256) {
    return yield* new CorpusError({ operation: "verify", message: `${file.path} expected ${file.sha256}, received ${actual}` })
  }
  yield* fs.makeDirectory(dirname(destination), { recursive: true }).pipe(
    Effect.mapError((error) => new CorpusError({ operation: "cache", message: String(error) })),
  )
  const temporary = `${destination}.${process.pid}.tmp`
  yield* fs.writeFile(temporary, bytes).pipe(
    Effect.zipRight(fs.rename(temporary, destination)),
    Effect.mapError((error) => new CorpusError({ operation: "cache", message: String(error) })),
  )
  return destination
})

const jsonLines = (text: string): Effect.Effect<readonly Record<string, unknown>[], CorpusError> =>
  Effect.forEach(
    text.split(/\r?\n/).filter((line) => line.trim().length > 0),
    (line, index) => Schema.decodeUnknown(Schema.parseJson(JsonObjectSchema))(line).pipe(
      Effect.mapError((error) => new CorpusError({
        operation: "parse-jsonl",
        message: `Invalid JSONL record ${index + 1}: ${String(error)}`,
      })),
    ),
  )

function normalizeSchema(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {}
  const input = value as Record<string, unknown>
  const output: Record<string, unknown> = {}
  for (const [key, item] of Object.entries(input)) {
    if (key === "type" && item === "any") continue
    if (key === "type" && item === "dict") output[key] = "object"
    else if (key === "type" && (item === "list" || item === "tuple")) output[key] = "array"
    else if (key === "type" && item === "float") output[key] = "number"
    else if (Array.isArray(item)) output[key] = item.map((entry) => typeof entry === "object" ? normalizeSchema(entry) : entry)
    else if (item && typeof item === "object") output[key] = normalizeSchema(item)
    else output[key] = item
  }
  return output
}

function toolsFrom(record: Record<string, unknown>): readonly ChatTool[] {
  const functions = Array.isArray(record.function) ? record.function : []
  return functions.flatMap((item) => {
    if (!item || typeof item !== "object") return []
    const fn = item as Record<string, unknown>
    if (typeof fn.name !== "string") return []
    return [{
      type: "function" as const,
      function: {
        name: fn.name,
        description: typeof fn.description === "string" ? fn.description : "BFCL function",
        parameters: normalizeSchema(fn.parameters) as ChatTool["function"]["parameters"],
      },
    }]
  })
}

function messagesFrom(record: Record<string, unknown>): readonly ChatMessage[] {
  const turns = Array.isArray(record.question) ? record.question : []
  const first = Array.isArray(turns[0]) ? turns[0] : turns
  return first.flatMap((item): ChatMessage[] => {
    if (!item || typeof item !== "object") return []
    const message = item as Record<string, unknown>
    if (message.role === "system" && typeof message.content === "string") return [{ role: "system", content: message.content }]
    if (message.role === "user" && typeof message.content === "string") return [{ role: "user", content: message.content }]
    return []
  })
}

function expectedFrom(record: Record<string, unknown>): readonly ExpectedToolCall[] {
  const groundTruth = Array.isArray(record.ground_truth) ? record.ground_truth : []
  return groundTruth.flatMap((entry): ExpectedToolCall[] => {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) return []
    return Object.entries(entry as Record<string, unknown>).map(([name, args]) => ({
      name,
      arguments: args && typeof args === "object" && !Array.isArray(args)
        ? Object.fromEntries(Object.entries(args as Record<string, unknown>).map(([key, values]) => [key, Array.isArray(values) ? values : [values]]))
        : {},
    }))
  })
}

function canonicalArguments(expected: ExpectedToolCall): Record<string, unknown> {
  return Object.fromEntries(Object.entries(expected.arguments).map(([key, allowed]) => [key, allowed[0]]))
}

function materialize(
  question: Record<string, unknown>,
  answer: Record<string, unknown>,
  lock: CorpusLock,
  questionFile: string,
  answerFile: string,
): Fixture | undefined {
  const id = typeof question.id === "string" ? question.id : undefined
  if (!id || answer.id !== id) return undefined
  const messages = messagesFrom(question)
  const tools = toolsFrom(question)
  const expected = expectedFrom(answer)
  if (messages.length === 0 || tools.length === 0 || expected.length === 0) return undefined
  const toolCalls = expected.map((call, index) => ({
    id: `bfcl_${id}_${index}`,
    type: "function" as const,
    function: { name: call.name, arguments: JSON.stringify(canonicalArguments(call)) },
  }))
  const canonicalAssistant: ChatMessage = { role: "assistant", content: null, tool_calls: toolCalls }
  const canonicalToolMessages: readonly ChatMessage[] = toolCalls.map((call) => ({
    role: "tool" as const,
    tool_call_id: call.id,
    content: JSON.stringify({ ok: true, tool: call.function.name, arguments: JSON.parse(call.function.arguments) }),
  }))
  return {
    id,
    messages,
    tools,
    expected,
    canonicalAssistant,
    canonicalToolMessages,
    provenance: { commit: lock.commit, questionFile, answerFile, recordId: id },
  }
}

export interface PrepareCorpusOptions {
  readonly root?: string
  readonly lockPath?: string
  readonly selectionPath?: string
  readonly offline?: boolean
}

export const prepareCorpus = (
  options: PrepareCorpusOptions = {},
): Effect.Effect<PreparedCorpus, CorpusError, FileSystem.FileSystem> => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const lock = yield* readJson(options.lockPath ?? defaultLockPath, CorpusLockSchema)
  const selection = yield* readJson(options.selectionPath ?? defaultSelectionPath, CorpusSelectionSchema)
  const root = options.root ?? join(cacheRoot(), lock.commit)
  const paths: Record<string, string> = {}
  for (const file of lock.files) {
    if (options.offline) {
      const path = join(root, file.path)
      const bytes = yield* fs.readFile(path).pipe(
        Effect.mapError((error) => new CorpusError({ operation: "verify", message: `${file.path}: ${String(error)}` })),
      )
      if (sha256(bytes) !== file.sha256) {
        return yield* new CorpusError({ operation: "verify", message: `${file.path} is absent or corrupt` })
      }
      paths[file.path] = path
    } else {
      paths[file.path] = yield* ensureFile(root, lock, file)
    }
  }
  const questionFiles = lock.files.filter((file) => !file.path.startsWith("possible_answer/"))
  if (questionFiles.length === 0) {
    return yield* new CorpusError({ operation: "select", message: "Lock contains no question files" })
  }
  const excluded = new Set(selection.exclusions)
  const cohorts = yield* Effect.forEach(questionFiles, ({ path: questionFile }) => Effect.gen(function* () {
    const answerFile = `possible_answer/${questionFile}`
    if (!paths[answerFile]) {
      return yield* new CorpusError({ operation: "select", message: `Missing answer lock for ${questionFile}` })
    }
    const questions = yield* jsonLines(yield* fs.readFileString(paths[questionFile]!))
    const answers = yield* jsonLines(yield* fs.readFileString(paths[answerFile]!))
    const answerById = new Map(answers.map((answer) => [String(answer.id), answer]))
    return questions
      .filter((question) => typeof question.id === "string" && selection.idPrefixes.some((prefix) => String(question.id).startsWith(prefix)) && !excluded.has(String(question.id)))
      .flatMap((question) => {
        const answer = answerById.get(String(question.id))
        if (!answer) return []
        const fixture = materialize(question, answer, lock, questionFile, answerFile)
        return fixture ? [fixture] : []
      })
  }).pipe(Effect.mapError((error) => error instanceof CorpusError
    ? error
    : new CorpusError({ operation: "materialize", message: String(error) }))))
  const maximumCohortLength = Math.max(...cohorts.map((cohort) => cohort.length))
  const fixtures = Array.from({ length: maximumCohortLength }, (_, index) => cohorts.flatMap((cohort) => cohort[index] ? [cohort[index]!] : []))
    .flat()
    .slice(0, selection.maximumRecords)
  if (fixtures.length === 0) {
    return yield* new CorpusError({ operation: "materialize", message: "No qualified BFCL fixtures were produced" })
  }
  const digest = digestObject({ lock, selection, fixtures })
  return { root, lock, selection, fixtures, digest }
})

export const corpusStatus = (
  options: PrepareCorpusOptions = {},
): Effect.Effect<readonly { readonly path: string; readonly valid: boolean }[], CorpusError, FileSystem.FileSystem> => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const lock = yield* readJson(options.lockPath ?? defaultLockPath, CorpusLockSchema)
  const root = options.root ?? join(cacheRoot(), lock.commit)
  return yield* Effect.forEach(lock.files, (file) => Effect.gen(function* () {
    const path = join(root, file.path)
    const bytes = yield* fs.readFile(path).pipe(Effect.option)
    return {
      path: file.path,
      valid: bytes._tag === "Some" && sha256(bytes.value) === file.sha256,
    }
  }))
})
