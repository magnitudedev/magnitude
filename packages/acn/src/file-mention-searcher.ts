import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as Path from "@effect/platform/Path"
import {
  AbsolutePathSchema,
  type DirectoryAccessDenied,
  type DirectoryCandidate,
  type DirectoryNotFound,
  type DirectoryPath,
  type FileAccessDenied,
  type FileNotFound,
  type FileSystemUnavailable,
  type InvalidDirectoryPath,
  type MentionAttachment,
  type MentionCandidate,
  type PathNotDirectory,
  type PathNotFile,
  type RawMentionOccurrence,
  type ReadFileResult,
  type ResolvePathResult,
  type SearchDirectoriesResult,
  type SearchMentionsResult,
  type WatchFileEvent,
} from "@magnitudedev/acn-protocol"
import { createId } from "@magnitudedev/generate-id"
import { resolveRgPath } from "@magnitudedev/ripgrep"
import { Chunk, Context, Data, Effect, Layer, Option, Stream } from "effect"
import { FileSystemManager } from "./file-system-manager"
import { GitInspector } from "./git-inspector"

export type TraversalError =
  | DirectoryNotFound
  | DirectoryAccessDenied
  | PathNotDirectory
  | FileSystemUnavailable

export type ReadHostFileError =
  | InvalidDirectoryPath
  | FileNotFound
  | FileAccessDenied
  | PathNotFile
  | FileSystemUnavailable

/** A client supplied an impossible mention span (ACN-internal, not wire). */
export class InvalidMentionPlacement extends Data.TaggedError("InvalidMentionPlacement")<{
  readonly start: number
  readonly end: number
}> {}

export interface FileMentionSearcher {
  readonly listFiles: (cwd: DirectoryPath, request: {
    readonly glob?: string | undefined
    readonly limit: number
  }) => Effect.Effect<ReadonlyArray<string>, TraversalError>
  readonly searchMentions: (cwd: DirectoryPath, request: {
    readonly query: string
    readonly limit: number
    readonly visibleLimit: number
    readonly includeRecent: boolean
  }) => Effect.Effect<SearchMentionsResult, TraversalError>
  readonly searchDirectories: (request: {
    readonly query: string
    readonly limit: number
    readonly includeRecent: boolean
    readonly recentDirectories: ReadonlyArray<{
      readonly path: string
      readonly lastActivity?: number | undefined
    }>
  }) => Effect.Effect<SearchDirectoriesResult>
  readonly readFile: (cwd: DirectoryPath, request: {
    readonly path: string
    readonly format: "text" | "base64"
    readonly offset: number
  }) => Effect.Effect<ReadFileResult, ReadHostFileError>
  readonly checkFileExists: (
    cwd: DirectoryPath,
    path: string,
  ) => Effect.Effect<boolean, InvalidDirectoryPath>
  readonly resolvePath: (cwd: DirectoryPath, request: {
    readonly path: string
    readonly checkExists: boolean
  }) => Effect.Effect<ResolvePathResult, InvalidDirectoryPath>
  readonly watchFile: (
    cwd: DirectoryPath,
    path: string,
  ) => Stream.Stream<WatchFileEvent, InvalidDirectoryPath>
  readonly collectMentionOccurrences: (input: {
    readonly cwd: DirectoryPath
    readonly scratchpadPath: string
    readonly content: string
    readonly provided: ReadonlyArray<RawMentionOccurrence>
  }) => Effect.Effect<ReadonlyArray<RawMentionOccurrence>, InvalidMentionPlacement>
}

export const FileMentionSearcher = Context.GenericTag<FileMentionSearcher>(
  "acn/FileMentionSearcher",
)

// ---------------------------------------------------------------------------
// Pure candidate helpers
// ---------------------------------------------------------------------------

const TRAILING_PUNCTUATION = new Set([".", ",", ";", "!", "?", ")", "]", "}"])

const BINARY_EXTENSIONS = new Set([
  ".exe", ".dll", ".so", ".o", ".pyc", ".class", ".jar", ".zip", ".tar", ".gz", ".bin", ".dat",
  ".db", ".sqlite", ".woff", ".woff2", ".ttf", ".eot", ".ico", ".mp3", ".mp4", ".mov", ".avi",
  ".pdf",
])

const IMAGE_EXTENSIONS = new Set([".png", ".jpg", ".jpeg", ".gif", ".webp"])

function globToRegExp(glob: string): RegExp {
  const escaped = glob
    .replace(/\*\*/g, "<<<GLOBSTAR>>>")
    .replace(/\./g, "\\.")
    .replace(/\*/g, "[^/]*")
    .replace(/<<<GLOBSTAR>>>/g, ".*")
    .replace(/\?/g, ".")
  return new RegExp(`^${escaped}$`)
}

function extensionOf(path: string): string {
  const name = path.slice(path.lastIndexOf("/") + 1)
  const dot = name.lastIndexOf(".")
  return dot <= 0 ? "" : name.slice(dot).toLowerCase()
}

function shouldKeepMentionPath(path: string): boolean {
  const extension = extensionOf(path)
  if (!extension) return true
  if (IMAGE_EXTENSIONS.has(extension)) return true
  return !BINARY_EXTENSIONS.has(extension)
}

function baseOf(path: string): string {
  const normalized = path.endsWith("/") ? path.slice(0, -1) : path
  const separator = normalized.lastIndexOf("/")
  return separator >= 0 ? normalized.slice(separator + 1) : normalized
}

function isSubsequence(query: string, text: string): boolean {
  if (!query) return true
  let queryIndex = 0
  let textIndex = 0
  while (queryIndex < query.length && textIndex < text.length) {
    if (query[queryIndex] === text[textIndex]) queryIndex++
    textIndex++
  }
  return queryIndex === query.length
}

function rankMentionPath(path: string, queryLower: string): number {
  const base = baseOf(path).toLowerCase()
  const full = path.toLowerCase()
  if (base.startsWith(queryLower)) return 0
  if (full.includes(queryLower)) return 1
  if (isSubsequence(queryLower, full)) return 2
  return 999
}

function collectMentionDirectories(paths: ReadonlyArray<string>): string[] {
  const directories = new Set<string>()
  for (const filePath of paths) {
    const parts = filePath.split("/").filter(Boolean)
    if (parts.length <= 1) continue
    let current = ""
    for (let index = 0; index < parts.length - 1; index++) {
      current += `${parts[index]}/`
      directories.add(current)
    }
  }
  return [...directories].sort((left, right) => left.localeCompare(right))
}

interface LineRange {
  readonly start: number
  readonly end: number
}

function expandLineRange(lineRange: LineRange): LineRange {
  if (lineRange.start !== lineRange.end) return lineRange
  return { start: Math.max(1, lineRange.start - 10), end: lineRange.end + 10 }
}

function parsePathAndRange(raw: string): { readonly path: string; readonly lineRange?: LineRange } {
  if (raw.endsWith(":")) return { path: raw.slice(0, -1) }
  const rangeMatch = raw.match(/:([\d]+)(?:-([\d]+))?$/)
  if (!rangeMatch || rangeMatch.index === 1 || rangeMatch[1] === undefined) return { path: raw }
  const start = Number.parseInt(rangeMatch[1], 10)
  const end = rangeMatch[2] !== undefined ? Number.parseInt(rangeMatch[2], 10) : start
  if (!Number.isFinite(start) || !Number.isFinite(end) || start < 1 || end < 1 || end < start) {
    return { path: raw }
  }
  return { path: raw.slice(0, rangeMatch.index), lineRange: expandLineRange({ start, end }) }
}

function toMentionFileCandidate(path: string, lineRange?: LineRange): MentionCandidate {
  return {
    path,
    kind: "file",
    contentType: "text",
    warning: false,
    ...(lineRange ? { lineRange } : {}),
  }
}

function toMentionDirectoryCandidate(path: string): MentionCandidate {
  return { path, kind: "directory", contentType: "directory", warning: false }
}

function isPathPrefix(query: string): boolean {
  return query.startsWith("/")
    || query.startsWith("~")
    || query.startsWith(".")
    || query.includes("/")
}

// ---------------------------------------------------------------------------
// Pure mention-text helpers
// ---------------------------------------------------------------------------

function looksFileLike(raw: string): boolean {
  return raw.includes("/")
    || raw.includes(".")
    || raw.includes("~")
    || /:\d+(?:-\d+)?$/.test(raw)
}

function maskCode(text: string): string {
  // Mention placements use JavaScript string offsets (UTF-16 code units).
  // split("") preserves that coordinate system; spreading would collapse
  // surrogate pairs and shift every span after an astral character.
  const chars = text.split("")
  let index = 0
  let inFence = false
  let lineStart = true

  while (index < chars.length) {
    if (lineStart) {
      const rest = chars.slice(index).join("")
      if (/^([ \t]*)(```|~~~)/.test(rest)) inFence = !inFence
    }
    if (inFence) {
      if (chars[index] !== "\n") chars[index] = " "
      lineStart = chars[index] === "\n"
      index++
      continue
    }
    if (chars[index] === "`") {
      const start = index
      index++
      while (index < chars.length && chars[index] !== "`") index++
      if (index < chars.length) {
        for (let masked = start; masked <= index; masked++) {
          if (chars[masked] !== "\n") chars[masked] = " "
        }
        index++
        continue
      }
    }
    lineStart = chars[index] === "\n"
    index++
  }
  return chars.join("")
}

function stripTrailingPunctuation(raw: string): string {
  let value = raw
  while (value.length > 0 && TRAILING_PUNCTUATION.has(value[value.length - 1] ?? "")) {
    value = value.slice(0, -1)
  }
  return value
}

interface InlineMentionCandidate {
  readonly raw: string
  readonly start: number
  readonly end: number
}

function extractInlineMentionCandidates(text: string): InlineMentionCandidate[] {
  const masked = maskCode(text)
  const candidates: InlineMentionCandidate[] = []
  const regex = /(^|[\s([{])@([^\s<>"'`]+)/g
  let match: RegExpExecArray | null
  while ((match = regex.exec(masked)) !== null) {
    const raw = match[2]
    if (!raw || !looksFileLike(raw)) continue
    const start = match.index + (match[1]?.length ?? 0)
    const stripped = stripTrailingPunctuation(raw)
    candidates.push({ raw, start, end: start + 1 + stripped.length })
  }
  return candidates
}

function expandExplicitMentionRange(mention: MentionAttachment): MentionAttachment {
  if (mention.type !== "mention_file_range") return mention
  const range = expandLineRange({ start: mention.startLine, end: mention.endLine })
  return { ...mention, startLine: range.start, endLine: range.end }
}

function overlaps(start: number, end: number, otherStart: number, otherEnd: number): boolean {
  return start < otherEnd && otherStart < end
}

type InlineOccurrence = RawMentionOccurrence & {
  readonly placement: { readonly _tag: "inline"; readonly start: number; readonly end: number }
}

const isInline = (occurrence: RawMentionOccurrence): occurrence is InlineOccurrence =>
  occurrence.placement._tag === "inline"

const validateProvidedOccurrences = (
  content: string,
  provided: ReadonlyArray<RawMentionOccurrence>,
): Effect.Effect<ReadonlyArray<RawMentionOccurrence>, InvalidMentionPlacement> =>
  Effect.gen(function* () {
    const inline = provided.filter(isInline).sort(
      (left, right) => left.placement.start - right.placement.start,
    )
    let previousEnd = -1
    for (const occurrence of inline) {
      const { start, end } = occurrence.placement
      if (
        !Number.isInteger(start) || !Number.isInteger(end)
        || start < 0 || end <= start || end > content.length
        || start < previousEnd
        || !content.slice(start, end).startsWith("@")
      ) {
        return yield* new InvalidMentionPlacement({ start, end })
      }
      previousEnd = end
    }
    return [...inline, ...provided.filter((item) => item.placement._tag === "trailing")]
  })

export const FileMentionSearcherLive: Layer.Layer<
  FileMentionSearcher,
  never,
  FileSystemManager | GitInspector | CommandExecutor.CommandExecutor | Path.Path
> = Layer.effect(
  FileMentionSearcher,
  Effect.gen(function* () {
    const fileSystem = yield* FileSystemManager
    const git = yield* GitInspector
    const executor = yield* CommandExecutor.CommandExecutor
    const paths = yield* Path.Path
    // Relative directory-picker queries resolve against the agent host cwd,
    // matching the existing product behavior of that host-wide surface.
    const hostBase = yield* fileSystem.normalizeDirectory(process.cwd()).pipe(Effect.orDie)

    const collect = <E>(stream: Stream.Stream<Uint8Array, E>): Effect.Effect<string, E> =>
      stream.pipe(
        Stream.decodeText(),
        Stream.runCollect,
        Effect.map((chunk) => Chunk.toReadonlyArray(chunk).join("")),
      )

    /** ripgrep is an optional accelerator: any failure selects the fallback. */
    const listFilesViaRipgrep = (
      cwd: DirectoryPath,
      limit: number,
    ): Effect.Effect<Option.Option<ReadonlyArray<string>>> =>
      Effect.gen(function* () {
        const rgPath = yield* Effect.tryPromise(() => resolveRgPath()).pipe(Effect.option)
        if (Option.isNone(rgPath)) return Option.none()
        const command = Command.make(
          rgPath.value,
          "--files",
          "-g", "!node_modules/**",
          "-g", "!dist/**",
          "-g", "!.git/**",
          "--max-count", String(limit),
          cwd,
        ).pipe(Command.workingDirectory(cwd))
        const outcome = yield* Effect.scoped(executor.start(command).pipe(
          Effect.flatMap((process) =>
            Effect.all([process.exitCode, collect(process.stdout)], { concurrency: 2 })),
        )).pipe(Effect.option)
        if (Option.isNone(outcome)) return Option.none()
        const [exitCode, stdout] = outcome.value
        if (Number(exitCode) !== 0) return Option.none()
        return Option.some(stdout
          .split("\n")
          .map((line) => line.trim())
          .filter((line) => line.length > 0)
          .map((line) => paths.relative(cwd, line)))
      })

    const listFiles = Effect.fn("acn.file-mention-searcher.list-files")(function* (
      cwd: DirectoryPath,
      request: { readonly glob?: string | undefined; readonly limit: number },
    ) {
      const viaRipgrep = yield* listFilesViaRipgrep(cwd, request.limit)
      const entries: ReadonlyArray<string> = Option.isSome(viaRipgrep)
        ? viaRipgrep.value
        : yield* fileSystem.openDirectory(cwd).pipe(
            Effect.flatMap((open) => open.walkFiles({ limit: request.limit })),
          )
      const filtered = request.glob !== undefined
        ? entries.filter((entry) => globToRegExp(request.glob ?? "").test(entry))
        : entries
      return filtered.slice(0, request.limit)
    })

    const recentGitFiles = (cwd: DirectoryPath, limit: number): Effect.Effect<ReadonlyArray<string>> =>
      git.recentFiles(cwd, limit).pipe(
        Effect.map((outcome) => outcome._tag === "recent_git_files" ? outcome.files : []),
      )

    const directoryLabel = (path: string): string => {
      const base = paths.basename(path)
      return base.length > 0 ? base : path
    }

    const isDirectory = (path: string): Effect.Effect<boolean> =>
      Effect.suspend(() => fileSystem.inspectPath(AbsolutePathSchema.make(path))).pipe(
        Effect.map((inspection) => inspection._tag === "directory"),
      )

    /** Lexical prefix containment (used for mention resolution allowances). */
    const isUnderPrefix = (absolute: string, prefix: string): boolean => {
      const relative = paths.relative(prefix, absolute)
      return relative === "" || (relative !== ".." && !relative.startsWith(`..${paths.sep}`))
    }

    const resolveInlineMention = (
      cwd: DirectoryPath,
      scratchpadPath: string,
      candidate: string,
    ): Effect.Effect<MentionAttachment | null> =>
      Effect.gen(function* () {
        const attempts = [candidate]
        const stripped = stripTrailingPunctuation(candidate)
        if (stripped !== candidate) attempts.push(stripped)

        for (const attempt of attempts) {
          const parsed = parsePathAndRange(attempt)
          if (!parsed.path || !looksFileLike(parsed.path)) continue
          const resolved = yield* fileSystem.resolveHostPath(cwd, parsed.path, {
            expandHome: true,
          }).pipe(Effect.option)
          if (Option.isNone(resolved)) continue
          const allowed = isUnderPrefix(resolved.value, cwd)
            || (scratchpadPath !== "" && isUnderPrefix(resolved.value, scratchpadPath))
          if (!allowed) continue
          const inspection = yield* fileSystem.inspectPath(resolved.value)
          if (inspection._tag === "directory") {
            return { type: "mention_directory", path: parsed.path }
          }
          if (inspection._tag === "file") {
            return parsed.lineRange !== undefined
              ? {
                  type: "mention_file_range",
                  path: parsed.path,
                  startLine: parsed.lineRange.start,
                  endLine: parsed.lineRange.end,
                }
              : { type: "mention_file", path: parsed.path }
          }
        }
        return null
      })

    return FileMentionSearcher.of({
      listFiles,

      searchMentions: Effect.fn("acn.file-mention-searcher.search-mentions")(function* (
        cwd,
        request,
      ) {
        const parsed = parsePathAndRange(request.query)
        const queryLower = parsed.path.toLowerCase()
        const fileLimit = Math.max(request.limit * 25, 1000)
        const files = (yield* listFiles(cwd, { limit: fileLimit })).filter(shouldKeepMentionPath)
        const directories = collectMentionDirectories(files)
        const allCandidates: MentionCandidate[] = [
          ...files.map((path) => toMentionFileCandidate(path, parsed.lineRange)),
          ...directories.map(toMentionDirectoryCandidate),
        ]

        if (!queryLower) {
          const recentPaths = request.includeRecent
            ? yield* recentGitFiles(cwd, Math.max(request.limit, request.visibleLimit))
            : []
          const indexed = new Set(files)
          const seen = new Set<string>()
          const recentCandidates: MentionCandidate[] = []
          for (const path of recentPaths) {
            if (!path || seen.has(path) || !indexed.has(path)) continue
            seen.add(path)
            recentCandidates.push(toMentionFileCandidate(path, parsed.lineRange))
            if (recentCandidates.length >= request.visibleLimit) break
          }
          const recentSet = new Set(recentCandidates.map((item) => item.path))
          const rest = allCandidates
            .filter((item) => !(item.kind === "file" && recentSet.has(item.path)))
            .sort((left, right) => left.path.localeCompare(right.path))
            .slice(0, request.limit)
          const ranked = [...recentCandidates, ...rest]
          const candidates = ranked.slice(0, request.visibleLimit)
          const visibleRecent = candidates.filter((item) => recentSet.has(item.path))
          return {
            query: parsed.path,
            ...(parsed.lineRange ? { lineRange: parsed.lineRange } : {}),
            candidates,
            recentCandidates: visibleRecent,
            overflowCount: Math.max(0, ranked.length - request.visibleLimit),
          }
        }

        const ranked = allCandidates
          .map((candidate) => ({ candidate, rank: rankMentionPath(candidate.path, queryLower) }))
          .filter((entry) => entry.rank < 999)
          .sort((left, right) =>
            (left.rank - right.rank) || left.candidate.path.localeCompare(right.candidate.path))
          .slice(0, request.limit)
          .map((entry) => entry.candidate)

        return {
          query: parsed.path,
          ...(parsed.lineRange ? { lineRange: parsed.lineRange } : {}),
          candidates: ranked.slice(0, request.visibleLimit),
          recentCandidates: [],
          overflowCount: Math.max(0, ranked.length - request.visibleLimit),
        }
      }),

      searchDirectories: Effect.fn("acn.file-mention-searcher.search-directories")(function* (
        request,
      ) {
        const trimmed = request.query.trim()
        const queryLower = trimmed.toLowerCase()
        const candidates: DirectoryCandidate[] = []
        const seen = new Set<string>()
        const push = (candidate: DirectoryCandidate) => {
          if (seen.has(candidate.path)) return
          seen.add(candidate.path)
          candidates.push(candidate)
        }

        if (trimmed && isPathPrefix(trimmed)) {
          const exact = yield* fileSystem.resolveHostPath(hostBase, trimmed, {
            expandHome: true,
          }).pipe(Effect.option)
          if (Option.isSome(exact) && (yield* isDirectory(exact.value))) {
            push({ path: exact.value, label: directoryLabel(exact.value), source: "exact" })
          }
        }

        if (request.includeRecent) {
          for (const recent of request.recentDirectories) {
            if (!recent.path) continue
            if (queryLower && !recent.path.toLowerCase().includes(queryLower)) continue
            push({
              path: recent.path,
              label: directoryLabel(recent.path),
              source: "recent",
              ...(recent.lastActivity !== undefined ? { lastActivity: recent.lastActivity } : {}),
            })
            if (candidates.length >= request.limit && !isPathPrefix(trimmed)) break
          }
        }

        if (candidates.length < request.limit && trimmed && isPathPrefix(trimmed)) {
          const expanded = yield* fileSystem.resolveHostPath(hostBase, trimmed, {
            expandHome: true,
          }).pipe(Effect.option)
          if (Option.isSome(expanded)) {
            const parent = trimmed.endsWith("/")
              ? expanded.value
              : AbsolutePathSchema.make(paths.dirname(expanded.value))
            const fragment = trimmed.endsWith("/")
              ? ""
              : paths.basename(expanded.value).toLowerCase()
            const listing = yield* fileSystem.listHostSubdirectories(parent)
            if (listing._tag === "listed") {
              const ranked = listing.names
                .map((name) => {
                  const nameLower = name.toLowerCase()
                  const rank = !fragment
                    ? 1
                    : nameLower.startsWith(fragment)
                      ? 0
                      : nameLower.includes(fragment)
                        ? 2
                        : 999
                  return { name, rank }
                })
                .filter((entry) => entry.rank < 999)
                .sort((left, right) =>
                  (left.rank - right.rank) || left.name.localeCompare(right.name))
                .slice(0, request.limit - candidates.length)
              for (const entry of ranked) {
                const path = paths.join(parent, entry.name)
                push({ path, label: directoryLabel(path), source: "filesystem" })
                if (candidates.length >= request.limit) break
              }
            }
          }
        }

        return { query: request.query, candidates: candidates.slice(0, request.limit) }
      }),

      readFile: Effect.fn("acn.file-mention-searcher.read-file")(function* (cwd, request) {
        const absolute = yield* fileSystem.resolveHostPath(cwd, request.path)
        const bytes = yield* fileSystem.readHostFile(absolute)
        const slice = request.offset > 0 ? bytes.subarray(Math.max(0, request.offset)) : bytes
        return {
          path: request.path,
          content: request.format === "base64"
            ? Buffer.from(slice).toString("base64")
            : new TextDecoder().decode(slice),
          format: request.format,
        }
      }),

      checkFileExists: Effect.fn("acn.file-mention-searcher.check-file-exists")(function* (
        cwd,
        path,
      ) {
        const absolute = yield* fileSystem.resolveHostPath(cwd, path)
        const inspection = yield* fileSystem.inspectPath(absolute)
        return inspection._tag === "file"
          || inspection._tag === "directory"
          || inspection._tag === "other"
      }),

      resolvePath: Effect.fn("acn.file-mention-searcher.resolve-path")(function* (cwd, request) {
        const absolute = yield* fileSystem.resolveHostPath(cwd, request.path)
        if (!request.checkExists) {
          return { resolved: absolute, exists: false, isDirectory: false }
        }
        const inspection = yield* fileSystem.inspectPath(absolute)
        return {
          resolved: absolute,
          exists: inspection._tag === "file"
            || inspection._tag === "directory"
            || inspection._tag === "other",
          isDirectory: inspection._tag === "directory",
        }
      }),

      watchFile: (cwd, path) => Stream.unwrap(
        fileSystem.resolveHostPath(cwd, path).pipe(
          Effect.map((absolute) => {
            const display = paths.relative(cwd, absolute) || path
            return fileSystem.watchHostFile(absolute).pipe(
              Stream.map((event): WatchFileEvent => ({ event: event._tag, path: display })),
            )
          }),
        ),
      ),

      collectMentionOccurrences: Effect.fn("acn.file-mention-searcher.collect-mention-occurrences")(
        function* (input) {
          const validated = (yield* validateProvidedOccurrences(input.content, input.provided))
            .map((occurrence) => ({
              ...occurrence,
              attachment: expandExplicitMentionRange(occurrence.attachment),
            }))
          const inline = validated.filter(isInline)
          const discovered: InlineOccurrence[] = []

          for (const candidate of extractInlineMentionCandidates(input.content)) {
            const taken = inline.some((occurrence) => overlaps(
              candidate.start,
              candidate.end,
              occurrence.placement.start,
              occurrence.placement.end,
            ))
            if (taken) continue
            const resolved = yield* resolveInlineMention(
              input.cwd,
              input.scratchpadPath,
              candidate.raw,
            )
            if (resolved === null) continue
            discovered.push({
              occurrenceId: createId(),
              attachment: resolved,
              placement: { _tag: "inline", start: candidate.start, end: candidate.end },
            })
          }

          const orderedInline: RawMentionOccurrence[] = [...inline, ...discovered]
            .sort((left, right) => left.placement.start - right.placement.start)
          const trailing = validated.filter((item) => item.placement._tag === "trailing")
          return [...orderedInline, ...trailing]
        },
      ),
    })
  }),
)
