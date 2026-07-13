import { readdir, stat } from "node:fs/promises"
import { basename, dirname, extname, resolve, relative, isAbsolute } from "node:path"
import { homedir } from "node:os"
import { Effect, Stream, Option, Ref, Schedule, Chunk } from "effect"
import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import { loadSkills } from "@magnitudedev/skills"
import { resolveRgPath } from "@magnitudedev/ripgrep"
import type {
  ReadFileResult,
  ResolvePathResult,
  MentionCandidate,
  DirectoryCandidate,
  SearchMentionsResult,
  SearchDirectoriesResult,
  RunBashResult,
  SessionError,
  SkillContent,
  SkillListEntry,
} from "@magnitudedev/protocol"
import { SessionOperationFailed } from "@magnitudedev/protocol"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function resolveSessionPath(requestedPath: string, cwd: string): string {
  if (isAbsolute(requestedPath)) return requestedPath
  return resolve(cwd, requestedPath)
}

function globToRegExp(glob: string): RegExp {
  const escaped = glob
    .replace(/\*\*/g, "<<<GLOBSTAR>>>")
    .replace(/\./g, "\\.")
    .replace(/\*/g, "[^/]*")
    .replace(/<<<GLOBSTAR>>>/g, ".*")
    .replace(/\?/g, ".")
  return new RegExp(`^${escaped}$`)
}

// ---------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------

const rgListFiles = (
  cwd: string,
  limit: number
): Effect.Effect<Array<string>, SessionError, CommandExecutor.CommandExecutor> =>
  Effect.gen(function* () {
    const rgPath = yield* Effect.tryPromise({
      try: () => resolveRgPath(),
      catch: (cause) =>
        new SessionOperationFailed({
          operation: "resolve ripgrep",
          reason: cause instanceof Error ? cause.message : "failed to resolve ripgrep",
        }),
    })
    const cmd = Command.make(
      rgPath,
      "--files",
      "-g",
      "!node_modules/**",
      "-g",
      "!dist/**",
      "-g",
      "!.git/**",
      "--max-count",
      String(limit),
      cwd
    ).pipe(Command.workingDirectory(cwd))
    const stdout = yield* Command.string(cmd).pipe(
      Effect.mapError(
        (cause) =>
          new SessionOperationFailed({
            operation: "list files",
            reason: cause instanceof Error ? cause.message : "ripgrep failed",
          })
      )
    )
    return stdout
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0)
      .map((line) => relative(cwd, line))
  })

const walkFiles = (
  cwd: string,
  limit: number
): Effect.Effect<Array<string>, SessionError, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const out: Array<string> = []
    const visit = (dir: string, prefix: string): Effect.Effect<void, SessionError> =>
      Effect.gen(function* () {
        if (out.length >= limit) return
        const names = yield* fs.readDirectory(dir).pipe(
          Effect.mapError(
            (cause) =>
              new SessionOperationFailed({
                operation: "list files",
                reason: cause instanceof Error ? cause.message : "read directory failed",
              })
          )
        )
        for (const name of names) {
          if (name.startsWith(".")) continue
          const fullPath = resolve(dir, name)
          const relativePath = prefix ? `${prefix}/${name}` : name
          const info = yield* fs.stat(fullPath).pipe(Effect.catchAll(() => Effect.succeed(null)))
          if (info === null) continue
          if (info.type === "Directory") {
            yield* visit(fullPath, relativePath)
          } else if (info.type === "File") {
            out.push(relativePath)
            if (out.length >= limit) return
          }
        }
      })
    yield* visit(cwd, "")
    return out
  })

export function listFiles(
  cwd: string,
  glob?: string | undefined,
  limit = 100
): Effect.Effect<Array<string>, SessionError, FileSystem.FileSystem | CommandExecutor.CommandExecutor> {
  return rgListFiles(cwd, limit).pipe(
    Effect.catchAll(() => walkFiles(cwd, limit)),
    Effect.map((entries) => {
      if (glob) {
        const pattern = globToRegExp(glob)
        return entries.filter((p) => pattern.test(p)).slice(0, limit)
      }
      return entries.slice(0, limit)
    })
  )
}

const BINARY_EXTENSIONS = new Set([
  ".exe", ".dll", ".so", ".o", ".pyc", ".class", ".jar", ".zip", ".tar", ".gz", ".bin", ".dat", ".db",
  ".sqlite", ".woff", ".woff2", ".ttf", ".eot", ".ico", ".mp3", ".mp4", ".mov", ".avi", ".pdf",
])

const IMAGE_EXTENSIONS = new Set([
  ".png", ".jpg", ".jpeg", ".gif", ".webp",
])

function shouldKeepMentionPath(path: string): boolean {
  const extension = extname(path).toLowerCase()
  if (!extension) return true
  if (IMAGE_EXTENSIONS.has(extension)) return true
  return !BINARY_EXTENSIONS.has(extension)
}

function mentionContentTypeFromPath(path: string): "text" {
  return "text"
}

function getBase(path: string): string {
  const normalized = path.endsWith("/") ? path.slice(0, -1) : path
  const i = normalized.lastIndexOf("/")
  return i >= 0 ? normalized.slice(i + 1) : normalized
}

function isSubsequence(query: string, text: string): boolean {
  if (!query) return true
  let qi = 0
  let ti = 0
  while (qi < query.length && ti < text.length) {
    if (query[qi] === text[ti]) qi++
    ti++
  }
  return qi === query.length
}

function rankMentionPath(path: string, queryLower: string): number {
  const base = getBase(path).toLowerCase()
  const full = path.toLowerCase()
  if (base.startsWith(queryLower)) return 0
  if (full.includes(queryLower)) return 1
  if (isSubsequence(queryLower, full)) return 2
  return 999
}

function collectMentionDirectories(paths: readonly string[]): string[] {
  const dirs = new Set<string>()
  for (const filePath of paths) {
    const parts = filePath.split("/").filter(Boolean)
    if (parts.length <= 1) continue
    let current = ""
    for (let i = 0; i < parts.length - 1; i++) {
      current += `${parts[i]}/`
      dirs.add(current)
    }
  }
  return [...dirs].sort((a, b) => a.localeCompare(b))
}

function expandMentionLineRange(lineRange: { start: number; end: number }): { start: number; end: number } {
  if (lineRange.start !== lineRange.end) return lineRange
  return { start: Math.max(1, lineRange.start - 10), end: lineRange.end + 10 }
}

function parseMentionPathAndRange(query: string): { filePath: string; lineRange?: { start: number; end: number } } {
  if (query.endsWith(":")) return { filePath: query.slice(0, -1) }

  const rangeMatch = query.match(/:([\d]+)(?:-([\d]+))?$/)
  if (!rangeMatch || rangeMatch.index === 1) return { filePath: query }

  const filePath = query.slice(0, rangeMatch.index)
  const start = parseInt(rangeMatch[1], 10)
  const end = rangeMatch[2] ? parseInt(rangeMatch[2], 10) : start

  if (start < 1 || end < 1 || end < start) return { filePath: query }

  return { filePath, lineRange: expandMentionLineRange({ start, end }) }
}

function toMentionFileCandidate(path: string, lineRange?: { start: number; end: number }): MentionCandidate {
  return {
    path,
    kind: "file",
    contentType: mentionContentTypeFromPath(path),
    warning: false,
    ...(lineRange ? { lineRange } : {}),
  }
}

function toMentionDirectoryCandidate(path: string): MentionCandidate {
  return {
    path,
    kind: "directory",
    contentType: "directory",
    warning: false,
  }
}

export function readFileOp(
  cwd: string,
  requestedPath: string,
  format: "text" | "base64",
  offset: number
): Effect.Effect<ReadFileResult, SessionError, FileSystem.FileSystem> {
  return Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const fullPath = resolveSessionPath(requestedPath, cwd)
    const buffer = yield* fs.readFile(fullPath).pipe(
      Effect.mapError(
        (cause) =>
          new SessionOperationFailed({
            operation: `read file ${requestedPath}`,
            reason: cause instanceof Error ? cause.message : "read failed",
          })
      )
    )
    let slice = buffer
    if (offset > 0) {
      const start = Math.max(0, offset)
      slice = buffer.subarray(start)
    }
    const content =
      format === "base64"
        ? Buffer.from(slice).toString("base64")
        : new TextDecoder().decode(slice)
    return {
      path: requestedPath,
      content,
      format,
    }
  })
}

export function checkFileExists(
  cwd: string,
  requestedPath: string
): Effect.Effect<boolean, never, FileSystem.FileSystem> {
  return Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const fullPath = resolveSessionPath(requestedPath, cwd)
    return yield* fs.exists(fullPath).pipe(Effect.catchAll(() => Effect.succeed(false)))
  })
}

export function watchFile(
  cwd: string,
  requestedPath: string
): Stream.Stream<{ event: "created" | "changed" | "removed"; path: string }, SessionError, FileSystem.FileSystem> {
  const fullPath = resolveSessionPath(requestedPath, cwd)
  const path = relative(cwd, fullPath) || requestedPath

  const watchNative = (fs: FileSystem.FileSystem) =>
    fs.watch(fullPath).pipe(
      Stream.map((event) => {
        if (event._tag === "Create") return { event: "created" as const, path }
        if (event._tag === "Update") return { event: "changed" as const, path }
        return { event: "removed" as const, path }
      }),
      Stream.mapError(
        (cause) =>
          new SessionOperationFailed({
            operation: `watch file ${requestedPath}`,
            reason: cause instanceof Error ? cause.message : "native watcher failed",
          })
      )
    )

  const pollForChanges = (fs: FileSystem.FileSystem) =>
    Stream.unwrap(
      Effect.gen(function* () {
        const previousRef = yield* Ref.make<Option.Option<{ size: bigint; mtimeMs: number }>>(Option.none())
        return Stream.repeatEffectWithSchedule(
          Effect.gen(function* () {
            const info = yield* fs.stat(fullPath).pipe(Effect.catchAll(() => Effect.succeed(null)))
            if (info === null) {
              const prev = yield* Ref.getAndSet(previousRef, Option.none())
              return Option.match(prev, {
                onNone: () => Option.none<{ event: "created" | "changed" | "removed"; path: string }>(),
                onSome: () => Option.some({ event: "removed" as const, path }),
              })
            }
            const current = {
              size: info.size,
              mtimeMs: Option.getOrUndefined(info.mtime)?.getTime() ?? 0,
            }
            const prev = yield* Ref.get(previousRef)
            const event = Option.match(prev, {
              onNone: () => Option.some({ event: "created" as const, path }),
              onSome: (p) =>
                p.size !== current.size || p.mtimeMs !== current.mtimeMs
                  ? Option.some({ event: "changed" as const, path })
                  : Option.none(),
            })
            yield* Ref.set(previousRef, Option.some(current))
            return event
          }),
          Schedule.spaced("500 millis")
        ).pipe(
          Stream.filter(Option.isSome),
          Stream.map((option) => option.value)
        )
      })
    )

  return Stream.unwrap(
    Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      return watchNative(fs).pipe(Stream.catchAll(() => pollForChanges(fs)))
    })
  )
}

export function resolvePath(
  cwd: string,
  requestedPath: string,
  checkExists = true
): Effect.Effect<ResolvePathResult, SessionError, FileSystem.FileSystem> {
  return Effect.gen(function* () {
    const resolved = resolveSessionPath(requestedPath, cwd)
    let exists = false
    let isDirectory = false
    if (checkExists) {
      const fs = yield* FileSystem.FileSystem
      const info = yield* fs.stat(resolved).pipe(Effect.catchAll(() => Effect.succeed(null)))
      if (info !== null) {
        exists = true
        isDirectory = info.type === "Directory"
      }
    }
    return { resolved, exists, isDirectory }
  })
}

export function searchMentions(
  cwd: string,
  query: string,
  limit = 40,
  visibleLimit = 10,
  includeRecent = true
): Effect.Effect<SearchMentionsResult, SessionError, FileSystem.FileSystem | CommandExecutor.CommandExecutor> {
  return Effect.gen(function* () {
    const parsed = parseMentionPathAndRange(query)
    const queryLower = parsed.filePath.toLowerCase()
    const fileLimit = Math.max(limit * 25, 1000)
    const files = (yield* listFiles(cwd, undefined, fileLimit)).filter(shouldKeepMentionPath)
    const directories = collectMentionDirectories(files)
    const allCandidates: MentionCandidate[] = [
      ...files.map((path) => toMentionFileCandidate(path, parsed.lineRange)),
      ...directories.map(toMentionDirectoryCandidate),
    ]

    if (!queryLower) {
      const recentPaths = includeRecent ? yield* getGitRecentFiles(cwd, Math.max(limit, visibleLimit)) : []
      const indexed = new Set(files)
      const seen = new Set<string>()
      const recentCandidates: MentionCandidate[] = []
      for (const path of recentPaths) {
        if (!path || seen.has(path) || !indexed.has(path)) continue
        seen.add(path)
        recentCandidates.push(toMentionFileCandidate(path, parsed.lineRange))
        if (recentCandidates.length >= visibleLimit) break
      }
      const recentSet = new Set(recentCandidates.map((item) => item.path))
      const rest = allCandidates
        .filter((item) => !(item.kind === "file" && recentSet.has(item.path)))
        .sort((a, b) => a.path.localeCompare(b.path))
        .slice(0, limit)
      const ranked = [...recentCandidates, ...rest]
      const candidates = ranked.slice(0, visibleLimit)
      const visibleRecent = candidates.filter((item) => recentSet.has(item.path))
      return {
        query: parsed.filePath,
        ...(parsed.lineRange ? { lineRange: parsed.lineRange } : {}),
        candidates,
        recentCandidates: visibleRecent,
        overflowCount: Math.max(0, ranked.length - visibleLimit),
      }
    }

    const ranked = allCandidates
      .map((candidate) => ({ candidate, rank: rankMentionPath(candidate.path, queryLower) }))
      .filter((entry) => entry.rank < 999)
      .sort((a, b) => (a.rank - b.rank) || a.candidate.path.localeCompare(b.candidate.path))
      .slice(0, limit)
      .map((entry) => entry.candidate)

    return {
      query: parsed.filePath,
      ...(parsed.lineRange ? { lineRange: parsed.lineRange } : {}),
      candidates: ranked.slice(0, visibleLimit),
      recentCandidates: [],
      overflowCount: Math.max(0, ranked.length - visibleLimit),
    }
  })
}

// ---------------------------------------------------------------------------
// Directories
// ---------------------------------------------------------------------------

interface RecentDirectoryInput {
  readonly path: string
  readonly lastActivity?: number
}

function expandHomePath(path: string): string {
  if (path === "~") return homedir()
  if (path.startsWith("~/")) return resolve(homedir(), path.slice(2))
  return path
}

function isPathPrefix(query: string): boolean {
  return query.startsWith("/")
    || query.startsWith("~")
    || query.startsWith(".")
    || query.includes("/")
}

function directoryLabel(path: string): string {
  const base = basename(path)
  return base.length > 0 ? base : path
}

async function directoryExists(path: string): Promise<boolean> {
  try {
    const info = await stat(path)
    return info.isDirectory()
  } catch {
    return false
  }
}

async function searchFilesystemDirectories(query: string, limit: number): Promise<DirectoryCandidate[]> {
  const trimmed = query.trim()
  if (!trimmed || !isPathPrefix(trimmed)) return []

  const expanded = expandHomePath(trimmed)
  const parent = trimmed.endsWith("/")
    ? expanded
    : dirname(expanded)
  const fragment = trimmed.endsWith("/") ? "" : basename(expanded).toLowerCase()

  if (!(await directoryExists(parent))) return []

  const entries = await readdir(parent, { withFileTypes: true })
  return entries
    .filter((entry) => entry.isDirectory())
    .filter((entry) => !entry.name.startsWith("."))
    .map((entry) => {
      const nameLower = entry.name.toLowerCase()
      const rank = !fragment
        ? 1
        : nameLower.startsWith(fragment)
          ? 0
          : nameLower.includes(fragment)
            ? 2
            : 999
      return { entry, rank }
    })
    .filter(({ rank }) => rank < 999)
    .sort((a, b) => (a.rank - b.rank) || a.entry.name.localeCompare(b.entry.name))
    .slice(0, limit)
    .map(({ entry }) => {
      const path = resolve(parent, entry.name)
      return {
        path,
        label: directoryLabel(path),
        source: "filesystem" as const,
      }
    })
}

export function searchDirectories(
  query: string,
  recentDirectories: readonly RecentDirectoryInput[],
  limit = 20,
  includeRecent = true
): Effect.Effect<SearchDirectoriesResult, SessionError> {
  return Effect.tryPromise({
    try: async () => {
      const trimmed = query.trim()
      const queryLower = trimmed.toLowerCase()
      const candidates: DirectoryCandidate[] = []
      const seen = new Set<string>()

      const push = (candidate: DirectoryCandidate) => {
        if (seen.has(candidate.path)) return
        seen.add(candidate.path)
        candidates.push(candidate)
      }

      if (trimmed && isPathPrefix(trimmed)) {
        const expanded = resolve(expandHomePath(trimmed))
        if (await directoryExists(expanded)) {
          push({
            path: expanded,
            label: directoryLabel(expanded),
            source: "exact",
          })
        }
      }

      if (includeRecent) {
        for (const recent of recentDirectories) {
          const path = recent.path
          if (!path) continue
          if (queryLower && !path.toLowerCase().includes(queryLower)) continue
          push({
            path,
            label: directoryLabel(path),
            source: "recent",
            ...(recent.lastActivity !== undefined ? { lastActivity: recent.lastActivity } : {}),
          })
          if (candidates.length >= limit && !isPathPrefix(trimmed)) break
        }
      }

      if (candidates.length < limit) {
        const filesystemCandidates = await searchFilesystemDirectories(trimmed, limit - candidates.length)
        for (const candidate of filesystemCandidates) {
          push(candidate)
          if (candidates.length >= limit) break
        }
      }

      return {
        query,
        candidates: candidates.slice(0, limit),
      }
    },
    catch: (cause) =>
      new SessionOperationFailed({
        operation: "search directories",
        reason: cause instanceof Error ? cause.message : "directory search failed",
      }),
  })
}

// ---------------------------------------------------------------------------
// Git
// ---------------------------------------------------------------------------

export function getGitRecentFiles(
  cwd: string,
  limit = 20
): Effect.Effect<Array<string>, never, CommandExecutor.CommandExecutor> {
  const cmd = Command.make(
    "git",
    "log",
    "--name-only",
    "--pretty=format:",
    "-n",
    String(limit * 2)
  ).pipe(Command.workingDirectory(cwd))
  return Command.string(cmd).pipe(
    Effect.map((text) => {
      const lines = text
        .split("\n")
        .map((l) => l.trim())
        .filter((l) => l.length > 0)
      const seen = new Set<string>()
      const result: Array<string> = []
      for (const line of lines) {
        if (seen.has(line)) continue
        seen.add(line)
        result.push(line)
        if (result.length >= limit) break
      }
      return result
    }),
    Effect.catchAll(() => Effect.succeed([]))
  )
}

// ---------------------------------------------------------------------------
// Skills
// ---------------------------------------------------------------------------

export function listSkills(
  cwd: string
): Effect.Effect<Array<SkillListEntry>, SessionError, FileSystem.FileSystem> {
  return Effect.gen(function* () {
    const skills = yield* Effect.tryPromise({
      try: () => loadSkills(cwd),
      catch: (cause) =>
        new SessionOperationFailed({
          operation: "list skills",
          reason: cause instanceof Error ? cause.message : "skill listing failed",
        }),
    })
    return Array.from(skills.values()).map((skill) => ({
      name: skill.name,
      description: skill.description,
      path: skill.path,
    }))
  })
}

export function getSkill(
  cwd: string,
  name: string
): Effect.Effect<SkillContent, SessionError, FileSystem.FileSystem> {
  return Effect.gen(function* () {
    const skills = yield* Effect.tryPromise({
      try: () => loadSkills(cwd),
      catch: (cause) =>
        new SessionOperationFailed({
          operation: `get skill ${name}`,
          reason: cause instanceof Error ? cause.message : "skill load failed",
        }),
    })
    const skill = skills.get(name)
    if (!skill) {
      return yield* new SessionOperationFailed({
        operation: `get skill ${name}`,
        reason: `Skill not found: ${name}`,
      })
    }
    const fs = yield* FileSystem.FileSystem
    const content = yield* fs.readFileString(skill.path).pipe(
      Effect.mapError(
        (cause) =>
          new SessionOperationFailed({
            operation: `get skill ${name}`,
            reason: cause instanceof Error ? cause.message : "skill read failed",
          })
      )
    )
    return { name: skill.name, content }
  })
}

// ---------------------------------------------------------------------------
// Bash
// ---------------------------------------------------------------------------

export function runBash(
  context: { cwd: string; projectRoot: string; scratchpadPath: string },
  command: string,
  stdin?: string | undefined
): Effect.Effect<RunBashResult, SessionError, CommandExecutor.CommandExecutor> {
  return Effect.scoped(
    Effect.gen(function* () {
      const shell = process.env.SHELL || "/bin/sh"
      const baseCmd = Command.make(shell, "-c", command).pipe(
        Command.workingDirectory(context.cwd),
        Command.env({
          ...process.env,
          PROJECT_ROOT: context.projectRoot,
          M: context.scratchpadPath,
        })
      )
      const cmd = stdin ? Command.feed(baseCmd, stdin) : baseCmd
      const proc = yield* Command.start(cmd)
      const stdout = yield* proc.stdout.pipe(
        Stream.decodeText(),
        Stream.runCollect,
        Effect.map((chunk) => Chunk.toReadonlyArray(chunk).join(""))
      )
      const stderr = yield* proc.stderr.pipe(
        Stream.decodeText(),
        Stream.runCollect,
        Effect.map((chunk) => Chunk.toReadonlyArray(chunk).join(""))
      )
      const exitCode = yield* proc.exitCode
      return {
        stdout,
        stderr,
        exitCode: Number(exitCode),
        cwd: context.cwd,
      }
    })
  ).pipe(
    Effect.mapError(
      (cause) =>
        new SessionOperationFailed({
          operation: "run bash",
          reason: cause instanceof Error ? cause.message : "bash failed",
        })
    )
  )
}
