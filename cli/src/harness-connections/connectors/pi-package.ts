import * as Command from "@effect/platform/Command"
import * as FileSystem from "@effect/platform/FileSystem"
import { Effect, Option, Schema, Stream } from "effect"
import { minimatch } from "minimatch"
import { basename, dirname, resolve, sep } from "node:path"
import { homedir } from "node:os"
import { isDeepStrictEqual } from "node:util"
import { satisfies } from "semver"
import type { HarnessCompanionPackage, HarnessCompanionState } from "../contract"
import type { HarnessConnectionPaths } from "../paths"
import { jsonObject, readOr, updateJsonc, writeIfChanged } from "../shared"
import { ConnectionTransaction } from "../transaction"
import { writeFileAtomic } from "../../utils/atomic-file"

export const PI_COMPANION_PACKAGE_NAME = "Magnitude for Pi"
export const PI_COMPANION_PACKAGE_IDENTITY = "@magnitudedev/pi"
export const PI_COMPANION_PACKAGE_SOURCE = "npm:@magnitudedev/pi@0.0.1"
export const PI_COMPANION_EXTENSION_PATH = "dist/magnitude.js"
const SUPPORTED_PACKAGE_VERSION = "0.0.1"

export class PiPackageError extends Schema.TaggedError<PiPackageError>()("PiPackageError", {
  message: Schema.String,
}) {}
const object = (value: unknown): Record<string, unknown> | undefined =>
  value !== null && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : undefined
const sourceOf = (entry: unknown): string | undefined => {
  const source = typeof entry === "string" ? entry : object(entry)?.source
  return typeof source === "string" ? source : undefined
}
const isCompanionNpm = (source: string) => source === `npm:${PI_COMPANION_PACKAGE_IDENTITY}`
  || source.startsWith(`npm:${PI_COMPANION_PACKAGE_IDENTITY}@`)
const localPath = (source: string, settings: string) => resolve(dirname(settings), source.startsWith("~/") ? `${homedir()}/${source.slice(2)}` : source)
const sameSource = (left: string, right: string, settings: string) =>
  left.startsWith("npm:") || right.startsWith("npm:") ? left === right : localPath(left, settings) === localPath(right, settings)
const posix = (path: string) => path.split(sep).join("/")
const exact = (pattern: string, root: string) => {
  const normalized = posix(pattern.startsWith("./") || pattern.startsWith(".\\") ? pattern.slice(2) : pattern)
  return normalized === PI_COMPANION_EXTENSION_PATH || normalized === posix(resolve(root, PI_COMPANION_EXTENSION_PATH))
}
const matches = (pattern: string, root: string) => [PI_COMPANION_EXTENSION_PATH, basename(PI_COMPANION_EXTENSION_PATH), posix(resolve(root, PI_COMPANION_EXTENSION_PATH))]
  .some((path) => minimatch(path, posix(pattern)))

/** Mirrors the supported Pi package manager's regular and autoload-disabled filters. */
export const piPackageExtensionEnabled = (entry: unknown, packageRoot = "/"): boolean => {
  if (typeof entry === "string") return true
  const config = object(entry)
  if (!config) return false
  const filters = config.extensions
  if (filters === undefined) return config.autoload !== false
  if (!Array.isArray(filters) || !filters.every((filter): filter is string => typeof filter === "string") || filters.length === 0) return false
  if (config.autoload === false) {
    let enabled = false
    for (const filter of filters) {
      const prefix = filter[0]
      const target = ["+", "-", "!"].includes(prefix!) ? filter.slice(1) : filter
      if (prefix === "+" || prefix === "-" ? exact(target, packageRoot) : matches(target, packageRoot)) enabled = prefix !== "-" && prefix !== "!"
    }
    return enabled
  }
  const includes = filters.filter((filter) => !["!", "+", "-"].includes(filter[0]!))
  let enabled = includes.length === 0 || includes.some((filter) => matches(filter, packageRoot))
  if (filters.some((filter) => filter.startsWith("!") && matches(filter.slice(1), packageRoot))) enabled = false
  if (filters.some((filter) => filter.startsWith("+") && exact(filter.slice(1), packageRoot))) enabled = true
  if (filters.some((filter) => filter.startsWith("-") && exact(filter.slice(1), packageRoot))) enabled = false
  return enabled
}

const PackageManifest = Schema.Struct({ name: Schema.String, version: Schema.String, pi: Schema.Struct({ extensions: Schema.Array(Schema.String) }) })
const decodeManifest = Schema.decodeUnknown(Schema.parseJson(PackageManifest))

export const makePiCompanion = (paths: HarnessConnectionPaths, desiredSource = PI_COMPANION_PACKAGE_SOURCE): HarnessCompanionPackage => {
  const agentDir = dirname(paths.piSettings)
  const find = (settings: Record<string, unknown>, source: string, byIdentity = false) => {
    const packages = Array.isArray(settings.packages) ? settings.packages : []
    const index = packages.findIndex((entry) => {
      const candidate = sourceOf(entry)
      return candidate !== undefined && (sameSource(candidate, source, paths.piSettings) || byIdentity && isCompanionNpm(candidate) && isCompanionNpm(source))
    })
    return index < 0 ? undefined : { index, entry: packages[index], source: sourceOf(packages[index])! }
  }
  const rootFor = (source: string) => source.startsWith("npm:")
    ? resolve(agentDir, "npm/node_modules", PI_COMPANION_PACKAGE_IDENTITY)
    : localPath(source, paths.piSettings)
  const command = (executable: string, action: "install" | "remove", source: string) => Effect.scoped(Effect.gen(function* () {
    const process = yield* Command.make(executable, action, source.startsWith("npm:") ? source : localPath(source, paths.piSettings)).pipe(
      Command.env({ PI_CODING_AGENT_DIR: agentDir }), Command.start,
    )
    const [exitCode, stdout, stderr] = yield* Effect.all([
      process.exitCode,
      readCommandOutput(process.stdout),
      readCommandOutput(process.stderr),
    ], { concurrency: "unbounded" })
    if (Number(exitCode) !== 0) return yield* new PiPackageError({ message: `Pi ${action} failed (${exitCode}): ${stderr || stdout}` })
  })).pipe(Effect.timeout("2 minutes"))
  const inspect = (source: string) => Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const root = rootFor(source)
    if (!(yield* fs.exists(resolve(root, "package.json")))) return false
    const manifest = yield* fs.readFileString(resolve(root, "package.json")).pipe(Effect.flatMap(decodeManifest))
    const configuredVersion = source.startsWith(`npm:${PI_COMPANION_PACKAGE_IDENTITY}@`) ? source.slice(`npm:${PI_COMPANION_PACKAGE_IDENTITY}@`.length) : "*"
    if (manifest.name !== PI_COMPANION_PACKAGE_IDENTITY || manifest.version !== SUPPORTED_PACKAGE_VERSION
      || source.startsWith("npm:") && !satisfies(manifest.version, configuredVersion)
      || !manifest.pi.extensions.includes(`./${PI_COMPANION_EXTENSION_PATH}`)) {
      return yield* new PiPackageError({ message: `Unsupported Magnitude for Pi package at ${root}. Expected version ${SUPPORTED_PACKAGE_VERSION}; update this package explicitly before connecting.` })
    }
    return yield* fs.exists(resolve(root, PI_COMPANION_EXTENSION_PATH))
  })
  const verifyHost = (executable: string) => Command.make(executable, "--version").pipe(Command.string, Effect.timeout("10 seconds"), Effect.flatMap((version) =>
    /^0\.84\.(?:[4-9]|[1-9]\d+)\s*$/.test(version.trim()) ? Effect.void
      : Effect.fail(new PiPackageError({ message: `Magnitude for Pi requires Pi 0.84.4–0.84.x; found ${version.trim()}.` }))))

  // Native package operations can fail after mutating settings or disk. Register the
  // inverse first, and never overwrite a changed package entry during recovery.
  const mutate = (executable: string, action: "install" | "remove", source: string) => Effect.gen(function* () {
    const tx = yield* ConnectionTransaction
    const beforeText = yield* readOr(paths.piSettings, "{}\n")
    const beforeSettings = jsonObject(beforeText)
    const before = find(beforeSettings, source)
    yield* tx.compensate(`Pi ${action} ${source}`, Effect.gen(function* () {
      const text = yield* readOr(paths.piSettings, "{}\n")
      const current = find(jsonObject(text), source, true)
      if (current && !sameSource(current.source, source, paths.piSettings)) return yield* new PiPackageError({ message: `Preserved changed Pi package source ${current.source}; manual recovery needed for ${source}.` })
      if (current && !isDeepStrictEqual(current.entry, before?.entry) && typeof current.entry !== "string") return yield* new PiPackageError({ message: `Preserved concurrently edited Pi package ${source}; manual recovery needed.` })
      yield* command(executable, action === "install" ? "remove" : "install", source)
      const after = yield* readOr(paths.piSettings, "{}\n")
      const settings = jsonObject(after)
      const packages = (Array.isArray(settings.packages) ? settings.packages : []).filter((entry) => {
        const candidate = sourceOf(entry)
        return candidate === undefined || !sameSource(candidate, source, paths.piSettings)
      })
      if (before) packages.splice(Math.min(before.index, packages.length), 0, before.entry)
      // This recovery write must not register new compensation.
      const restored = updateJsonc(after, [[["packages"], packages.length === 0 && beforeSettings.packages === undefined ? undefined : packages]])
      yield* writeFileAtomic(paths.piSettings, isDeepStrictEqual(jsonObject(restored), beforeSettings) ? beforeText : restored)
    }))
    yield* command(executable, action, source)
  })

  return {
    description: { name: PI_COMPANION_PACKAGE_NAME, source: desiredSource, securityNotice: "Pi extensions execute with your user permissions." },
    activationInstructions: Option.some("Restart existing Pi sessions or run /reload to activate the extension."),
    reconcile: ({ installation, previous }) => Effect.gen(function* () {
      yield* verifyHost(installation.executable)
      let text = yield* readOr(paths.piSettings, "{}\n")
      let current = find(jsonObject(text), desiredSource, true)
        ?? (Option.isSome(previous) ? find(jsonObject(text), previous.value.source) : undefined)
      let owned = Option.isSome(previous) && previous.value.ownership === "magnitude"
        && (current === undefined || sameSource(current.source, previous.value.source, paths.piSettings))
      let installed = false
      if (owned && current && !sameSource(current.source, desiredSource, paths.piSettings)) {
        yield* mutate(installation.executable, "remove", current.source)
        current = undefined
      }
      const configuredSource = current?.source ?? desiredSource
      const source = configuredSource.startsWith("npm:") ? configuredSource : localPath(configuredSource, paths.piSettings)
      if (!current || !(yield* inspect(source))) {
        if (!current) owned = true
        yield* mutate(installation.executable, "install", source)
        if (!(yield* inspect(source))) return yield* new PiPackageError({ message: `Pi did not install a usable Magnitude extension at ${rootFor(source)}.` })
        installed = true
        text = yield* readOr(paths.piSettings, "{}\n")
        current = find(jsonObject(text), source)
        if (!current) return yield* new PiPackageError({ message: `Pi did not register ${source}.` })
      }
      const enabled = piPackageExtensionEnabled(current.entry, rootFor(source))
      let enablement = Option.isSome(previous) && previous.value.ownership === "pre-existing"
        && sameSource(previous.value.source, source, paths.piSettings) ? previous.value.enablement : Option.none()
      if (!enabled) {
        const config = object(current.entry) ?? { source }
        if (config.extensions !== undefined) yield* Schema.decodeUnknown(Schema.Array(Schema.String))(config.extensions)
        const filters = Array.isArray(config.extensions) ? config.extensions as string[] : []
        const after = [...filters.filter((filter) => !((filter.startsWith("+") || filter.startsWith("-")) && exact(filter.slice(1), rootFor(source)))), `+${PI_COMPANION_EXTENSION_PATH}`]
        enablement = Option.some({ before: Option.fromNullable(config.extensions as string[] | undefined), after })
        yield* writeIfChanged(paths.piSettings, text, updateJsonc(text, [[["packages", current.index, "extensions"], after]]))
      }
      const state: HarnessCompanionState = owned
        ? { identity: PI_COMPANION_PACKAGE_IDENTITY, source, ownership: "magnitude" }
        : { identity: PI_COMPANION_PACKAGE_IDENTITY, source, ownership: "pre-existing", enablement }
      return { state, status: installed ? "installed" : enabled ? "already-installed" : "enabled" }
    }),
    disconnect: ({ installation, state }) => Effect.gen(function* () {
      const text = yield* readOr(paths.piSettings, "{}\n")
      const current = find(jsonObject(text), state.source)
      if (!current) return
      if (state.ownership === "magnitude") { yield* verifyHost(installation.executable); yield* mutate(installation.executable, "remove", state.source); return }
      if (Option.isNone(state.enablement)) return
      const { before, after } = state.enablement.value
      if (!isDeepStrictEqual(object(current.entry)?.extensions, after)) return
      yield* writeIfChanged(paths.piSettings, text, updateJsonc(text, [[["packages", current.index, "extensions"], Option.getOrUndefined(before)]]))
    }),
  }
}

const readCommandOutput = <E, R>(stream: Stream.Stream<Uint8Array, E, R>) => stream.pipe(Stream.decodeText(), Stream.runFold("", (text, chunk) => (text + chunk).slice(-8_192)))
