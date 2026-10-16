import { FileSystem } from "@effect/platform"
import type { PlatformError } from "@effect/platform/Error"
import { Effect, Either, Option, Schema } from "effect"
import { homedir } from "node:os"
import { posix, win32 } from "node:path"
import type { AutomaticModelLocationKind, ModelLocationState } from "@magnitudedev/acn-protocol"

/** The host facts automatic model locations depend on; injected so every platform's rules are testable. */
export interface DiscoveryEnvironment {
  readonly platform: NodeJS.Platform
  readonly env: Readonly<Record<string, string | undefined>>
  readonly home: string
}

export const hostDiscoveryEnvironment = (): DiscoveryEnvironment => ({
  platform: process.platform,
  env: process.env,
  home: homedir(),
})

/** One automatic location's candidate directories, most authoritative first. */
export interface AutomaticLocationCandidates {
  readonly kind: AutomaticModelLocationKind
  readonly candidates: readonly [string, ...string[]]
}

/** The directory Magnitude scans for one automatic location, and whether it exists and can be read. */
export interface ResolvedAutomaticLocation {
  readonly kind: AutomaticModelLocationKind
  readonly path: string
  readonly state: ModelLocationState
}

const pathFor = (platform: NodeJS.Platform) => platform === "win32" ? win32 : posix

const configured = (value: string | undefined): Option.Option<string> =>
  Option.fromNullable(value?.trim()).pipe(Option.filter((trimmed) => trimmed.length > 0))

/** Expands `~` and makes the path absolute for the host platform. */
export const expandPath = (environment: DiscoveryEnvironment, value: string): string => {
  const path = pathFor(environment.platform)
  const expanded = value === "~"
    ? environment.home
    : value.startsWith("~/") || value.startsWith("~\\")
      ? path.join(environment.home, value.slice(2))
      : value
  return path.isAbsolute(expanded) ? path.normalize(expanded) : path.resolve(environment.home, expanded)
}

/** `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `HF_HOME/hub`, `XDG_CACHE_HOME/huggingface/hub`, then `~/.cache/huggingface/hub`. */
export const huggingFaceCacheCandidates = (environment: DiscoveryEnvironment): readonly [string, ...string[]] => {
  const path = pathFor(environment.platform)
  const { env } = environment
  const root = Option.firstSomeOf([
    configured(env.HF_HUB_CACHE),
    configured(env.HUGGINGFACE_HUB_CACHE),
    Option.map(configured(env.HF_HOME), (home) => path.join(home, "hub")),
    Option.map(configured(env.XDG_CACHE_HOME), (cache) => path.join(cache, "huggingface", "hub")),
  ]).pipe(Option.getOrElse(() => path.join(environment.home, ".cache", "huggingface", "hub")))
  return [expandPath(environment, root)]
}

/**
 * LM Studio keeps its models folder in its own settings (`downloadsFolder`), under its home
 * (`~/.lmstudio`, relocatable through `~/.lmstudio-home-pointer`); older installs used
 * `~/.cache/lm-studio`.
 */
export const lmStudioCandidates = (
  environment: DiscoveryEnvironment,
  settings: { readonly homePointer: Option.Option<string>; readonly downloadsFolder: Option.Option<string> },
): readonly [string, ...string[]] => {
  const path = pathFor(environment.platform)
  const home = Option.getOrElse(
    Option.map(settings.homePointer, (pointer) => expandPath(environment, pointer)),
    () => path.join(environment.home, ".lmstudio"),
  )
  const defaults = [path.join(home, "models"), path.join(environment.home, ".cache", "lm-studio", "models")]
  return Option.match(settings.downloadsFolder, {
    onNone: () => [defaults[0]!, defaults[1]!],
    onSome: (folder) => [expandPath(environment, folder), ...defaults],
  })
}

/** `LLAMA_CACHE`, then the platform cache: macOS `~/Library/Caches`, Linux XDG or `~/.cache`, Windows `%LOCALAPPDATA%`. */
export const llamaCppCandidates = (environment: DiscoveryEnvironment): readonly [string, ...string[]] => {
  const path = pathFor(environment.platform)
  const { env } = environment
  const root = Option.getOrElse(configured(env.LLAMA_CACHE), () => {
    switch (environment.platform) {
      case "darwin": return path.join(environment.home, "Library", "Caches", "llama.cpp")
      case "win32": return path.join(
        Option.getOrElse(configured(env.LOCALAPPDATA), () => path.join(environment.home, "AppData", "Local")),
        "llama.cpp",
      )
      default: return path.join(
        Option.getOrElse(configured(env.XDG_CACHE_HOME), () => path.join(environment.home, ".cache")),
        "llama.cpp",
      )
    }
  })
  return [expandPath(environment, root)]
}

/**
 * `OLLAMA_MODELS`, then `~/.ollama/models`. On Linux a system service keeps its models under the
 * `ollama` user's home, `/usr/share/ollama/.ollama/models` (the official install script), or directly
 * in `/var/lib/ollama` (distribution packages such as Arch's, which set `OLLAMA_MODELS` there).
 */
export const ollamaCandidates = (environment: DiscoveryEnvironment): readonly [string, ...string[]] => {
  const path = pathFor(environment.platform)
  const override = configured(environment.env.OLLAMA_MODELS)
  if (Option.isSome(override)) return [expandPath(environment, override.value)]
  const user = path.join(environment.home, ".ollama", "models")
  return environment.platform === "linux" ? [user, "/usr/share/ollama/.ollama/models", "/var/lib/ollama"] : [user]
}

export const automaticLocationCandidates = (
  environment: DiscoveryEnvironment,
  lmStudio: { readonly homePointer: Option.Option<string>; readonly downloadsFolder: Option.Option<string> },
): readonly AutomaticLocationCandidates[] => [
  { kind: "HuggingFaceCache", candidates: huggingFaceCacheCandidates(environment) },
  { kind: "LmStudio", candidates: lmStudioCandidates(environment, lmStudio) },
  { kind: "LlamaCpp", candidates: llamaCppCandidates(environment) },
  { kind: "Ollama", candidates: ollamaCandidates(environment) },
]

const LmStudioSettings = Schema.Struct({ downloadsFolder: Schema.optional(Schema.String) })

const readOptional = (fs: FileSystem.FileSystem, file: string) =>
  fs.readFileString(file).pipe(Effect.option)

/** Reads LM Studio's own home pointer and settings; a missing or unreadable file means its defaults. */
export const readLmStudioSettings = (environment: DiscoveryEnvironment) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const path = pathFor(environment.platform)
  const homePointer = (yield* readOptional(fs, path.join(environment.home, ".lmstudio-home-pointer"))).pipe(
    Option.flatMap((text) => configured(text)),
  )
  const home = Option.getOrElse(
    Option.map(homePointer, (pointer) => expandPath(environment, pointer)),
    () => path.join(environment.home, ".lmstudio"),
  )
  const settingsFiles = [path.join(home, "settings.json"), path.join(environment.home, ".cache", "lm-studio", "settings.json")]
  let downloadsFolder = Option.none<string>()
  for (const file of settingsFiles) {
    const text = yield* readOptional(fs, file)
    downloadsFolder = text.pipe(
      Option.flatMap(Schema.decodeUnknownOption(Schema.parseJson(LmStudioSettings))),
      Option.flatMap(({ downloadsFolder: folder }) => configured(folder)),
    )
    if (Option.isSome(downloadsFolder)) break
  }
  return { homePointer, downloadsFolder }
})

const isPermissionDenied = (error: PlatformError) => error._tag === "SystemError" && error.reason === "PermissionDenied"

/**
 * `Found` for a directory Magnitude can list, `Unreadable` when the directory or a folder above it
 * denies Magnitude permission, and `NotFound` otherwise.
 */
export const directoryState = (path: string): Effect.Effect<ModelLocationState, never, FileSystem.FileSystem> => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const stat = yield* fs.stat(path).pipe(Effect.either)
  if (Either.isLeft(stat)) return isPermissionDenied(stat.left) ? "Unreadable" : "NotFound"
  if (stat.right.type !== "Directory") return "NotFound"
  const listing = yield* fs.readDirectory(path).pipe(Effect.either)
  return Either.isRight(listing) || !isPermissionDenied(listing.left) ? "Found" : "Unreadable"
})

/** Each automatic location's first found candidate, else its first unreadable one, else its first candidate. */
export const resolveAutomaticLocations = (environment: DiscoveryEnvironment) => Effect.gen(function* () {
  const lmStudio = yield* readLmStudioSettings(environment)
  return yield* Effect.forEach(automaticLocationCandidates(environment, lmStudio), ({ kind, candidates }) =>
    Effect.gen(function* () {
      const states = yield* Effect.forEach(candidates, (path) =>
        Effect.map(directoryState(path), (state): ResolvedAutomaticLocation => ({ kind, path, state })))
      const fallback: ResolvedAutomaticLocation = { kind, path: candidates[0], state: "NotFound" }
      return states.find(({ state }) => state === "Found")
        ?? states.find(({ state }) => state === "Unreadable")
        ?? fallback
    }))
})
