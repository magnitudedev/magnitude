import { FileSystem } from "@effect/platform"
import type { PlatformError } from "@effect/platform/Error"
import { Array, Effect, Either, Option, Schema } from "effect"
import { homedir } from "node:os"
import { posix, win32 } from "node:path"
import {
  modelLocationState,
  type AutomaticModelLocationKind,
  type ModelLocationPath,
  type ModelLocationState,
} from "@magnitudedev/acn-protocol"

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

type Candidates = readonly [string, ...string[]]

/** One automatic location's candidate directories, most authoritative first. */
export interface AutomaticLocationCandidates {
  readonly kind: AutomaticModelLocationKind
  readonly candidates: Candidates
}

/**
 * Every distinct directory of one automatic location, each with whether it exists and can be read,
 * and the location's state across them. Magnitude scans all of them.
 */
export interface ResolvedAutomaticLocation {
  readonly kind: AutomaticModelLocationKind
  readonly paths: readonly [ModelLocationPath, ...ModelLocationPath[]]
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

/** The configured directories, then the defaults, expanded, each once, in order. */
const distinctCandidates = (
  environment: DiscoveryEnvironment,
  configuredPaths: readonly Option.Option<string>[],
  defaults: Candidates,
): Candidates => {
  const expanded = [...Array.getSomes(configuredPaths), ...defaults].map((value) => expandPath(environment, value))
  const distinct = expanded.filter((value, index) => expanded.indexOf(value) === index)
  return [distinct[0]!, ...distinct.slice(1)]
}

/**
 * `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `HF_HOME/hub` and `XDG_CACHE_HOME/huggingface/hub`, then
 * the default `~/.cache/huggingface/hub`, which tools run without those variables still fill.
 */
export const huggingFaceCacheCandidates = (environment: DiscoveryEnvironment): Candidates => {
  const path = pathFor(environment.platform)
  const { env } = environment
  return distinctCandidates(environment, [
    configured(env.HF_HUB_CACHE),
    configured(env.HUGGINGFACE_HUB_CACHE),
    Option.map(configured(env.HF_HOME), (home) => path.join(home, "hub")),
    Option.map(configured(env.XDG_CACHE_HOME), (cache) => path.join(cache, "huggingface", "hub")),
  ], [path.join(environment.home, ".cache", "huggingface", "hub")])
}

/**
 * LM Studio keeps its models folder in its own settings (`downloadsFolder`), under its home
 * (`~/.lmstudio`, relocatable through `~/.lmstudio-home-pointer`); older installs used
 * `~/.cache/lm-studio`, whose models stay there after an upgrade.
 */
export const lmStudioCandidates = (
  environment: DiscoveryEnvironment,
  settings: { readonly homePointer: Option.Option<string>; readonly downloadsFolder: Option.Option<string> },
): Candidates => {
  const path = pathFor(environment.platform)
  const home = Option.getOrElse(
    Option.map(settings.homePointer, (pointer) => expandPath(environment, pointer)),
    () => path.join(environment.home, ".lmstudio"),
  )
  return distinctCandidates(environment, [settings.downloadsFolder],
    [path.join(home, "models"), path.join(environment.home, ".cache", "lm-studio", "models")])
}

/**
 * `LLAMA_CACHE`, then the platform cache: macOS `~/Library/Caches`, Windows `%LOCALAPPDATA%`, and on
 * Linux `XDG_CACHE_HOME` and `~/.cache`.
 */
export const llamaCppCandidates = (environment: DiscoveryEnvironment): Candidates => {
  const path = pathFor(environment.platform)
  const { env } = environment
  const inCache = (cache: string) => path.join(cache, "llama.cpp")
  switch (environment.platform) {
    case "darwin":
      return distinctCandidates(environment, [configured(env.LLAMA_CACHE)],
        [inCache(path.join(environment.home, "Library", "Caches"))])
    case "win32":
      return distinctCandidates(environment, [configured(env.LLAMA_CACHE)], [inCache(
        Option.getOrElse(configured(env.LOCALAPPDATA), () => path.join(environment.home, "AppData", "Local")),
      )])
    default:
      return distinctCandidates(environment,
        [configured(env.LLAMA_CACHE), Option.map(configured(env.XDG_CACHE_HOME), inCache)],
        [inCache(path.join(environment.home, ".cache"))])
  }
}

/**
 * `OLLAMA_MODELS`, then `~/.ollama/models`, which a user's own `ollama serve` fills. On Linux a system
 * service keeps its models apart, under the `ollama` user's home, `/usr/share/ollama/.ollama/models`
 * (the official install script), or directly in `/var/lib/ollama` (distribution packages such as
 * Arch's, which set `OLLAMA_MODELS` there); a machine often has both a user and a system install.
 */
export const ollamaCandidates = (environment: DiscoveryEnvironment): Candidates => {
  const path = pathFor(environment.platform)
  const user = path.join(environment.home, ".ollama", "models")
  return distinctCandidates(environment, [configured(environment.env.OLLAMA_MODELS)],
    environment.platform === "linux" ? [user, "/usr/share/ollama/.ollama/models", "/var/lib/ollama"] : [user])
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

/**
 * The candidates that are distinct directories: one reached through a symlink, such as
 * `/usr/share/ollama` linked to `/var/lib/ollama`, is the same directory as its target, and the
 * first spelling is kept.
 */
const canonicalCandidates = (candidates: Candidates) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const canonical = yield* Effect.forEach(candidates, (path) => fs.realPath(path).pipe(Effect.orElseSucceed(() => path)))
  const distinct = candidates.filter((_, index) => canonical.indexOf(canonical[index]!) === index)
  return [distinct[0]!, ...distinct.slice(1)] as const
})

/** Every distinct candidate directory of each automatic location, with its state. */
export const resolveAutomaticLocations = (environment: DiscoveryEnvironment) => Effect.gen(function* () {
  const lmStudio = yield* readLmStudioSettings(environment)
  return yield* Effect.forEach(automaticLocationCandidates(environment, lmStudio), ({ kind, candidates }) =>
    Effect.gen(function* () {
      const [first, ...rest] = yield* canonicalCandidates(candidates)
      const resolvePath = (path: string) => Effect.map(directoryState(path), (state): ModelLocationPath => ({ path, state }))
      const paths = [yield* resolvePath(first), ...yield* Effect.forEach(rest, resolvePath)] as const
      return { kind, paths, state: modelLocationState(paths) } satisfies ResolvedAutomaticLocation
    }))
})
