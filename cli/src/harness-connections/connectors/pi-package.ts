import * as Command from "@effect/platform/Command"
import * as FileSystem from "@effect/platform/FileSystem"
import { Array, Effect, Option, Schema, Stream, pipe } from "effect"
import { minimatch } from "minimatch"
import { basename, dirname, resolve, sep } from "node:path"
import { homedir } from "node:os"
import { isDeepStrictEqual } from "node:util"
import { satisfies } from "semver"
import { MAGNITUDE_RPC_VERSION } from "@magnitudedev/sdk"
import releasePlan from "@magnitudedev/release/plan"
import { verifyPluginContent } from "@magnitudedev/release/plugin-content"
import type { HarnessCompanionPackage, HarnessCompanionState } from "../contract"
import type { HarnessConnectionPaths } from "../paths"
import { updateJsonc, writeIfChanged } from "../shared"
import { ConnectionTransaction } from "../transaction"
import { writeFileAtomic } from "@magnitudedev/utils/atomic-file"
import { PiPackageSourceSchema, type PiPackageSource } from "./pi-package-state"
import {
  decodePiSettings, piPackageFilters, piPackageSource, readPiSettings, replacePiPackages,
  type PiPackageEntry, type PiSettings,
} from "./pi-settings"

export const PI_COMPANION_PACKAGE_NAME = "Magnitude for Pi"
export const PI_COMPANION_PACKAGE_IDENTITY = "@magnitudedev/pi-extension"
const selectedPackage = Option.getOrThrowWith(
  Array.findFirst(releasePlan.plugins, ({ artifact }) => artifact.host === "pi"),
  () => new TypeError("Release preparation omitted the Pi plugin selection"),
).artifact
export const PI_COMPANION_PACKAGE_SOURCE = PiPackageSourceSchema.make(`npm:${selectedPackage.name}@${selectedPackage.version}`)
export const PI_COMPANION_EXTENSION_PATH = "dist/magnitude.js"

export class PiPackageError extends Schema.TaggedError<PiPackageError>()("PiPackageError", {
  message: Schema.String,
}) {}
const isCompanionNpm = (source: PiPackageSource) => source === `npm:${PI_COMPANION_PACKAGE_IDENTITY}`
  || source.startsWith(`npm:${PI_COMPANION_PACKAGE_IDENTITY}@`)
const localPath = (source: PiPackageSource, settings: string) => PiPackageSourceSchema.make(
  resolve(dirname(settings), source.startsWith("~/") ? `${homedir()}/${source.slice(2)}` : source),
)
const sameSource = (left: PiPackageSource, right: PiPackageSource, settings: string) =>
  left.startsWith("npm:") || right.startsWith("npm:") ? left === right : localPath(left, settings) === localPath(right, settings)
const posix = (path: string) => path.split(sep).join("/")
const exact = (pattern: string, root: string) => {
  const normalized = posix(pattern.startsWith("./") || pattern.startsWith(".\\") ? pattern.slice(2) : pattern)
  return normalized === PI_COMPANION_EXTENSION_PATH || normalized === posix(resolve(root, PI_COMPANION_EXTENSION_PATH))
}
const matches = (pattern: string, root: string) => [PI_COMPANION_EXTENSION_PATH, basename(PI_COMPANION_EXTENSION_PATH), posix(resolve(root, PI_COMPANION_EXTENSION_PATH))]
  .some((path) => minimatch(path, posix(pattern)))

/** Mirrors the supported Pi package manager's regular and autoload-disabled filters. */
export const piPackageExtensionEnabled = (entry: PiPackageEntry, packageRoot = "/"): boolean => {
  if (typeof entry === "string") return true
  const autoload = Option.getOrElse(entry.autoload, () => true)
  if (Option.isNone(entry.extensions)) return autoload
  const filters = entry.extensions.value
  if (filters.length === 0) return false
  if (!autoload) {
    let enabled = false
    for (const filter of filters) {
      const prefix = filter.charAt(0)
      const target = ["+", "-", "!"].includes(prefix) ? filter.slice(1) : filter
      if (prefix === "+" || prefix === "-" ? exact(target, packageRoot) : matches(target, packageRoot)) enabled = prefix !== "-" && prefix !== "!"
    }
    return enabled
  }
  const includes = filters.filter((filter) => !["!", "+", "-"].includes(filter.charAt(0)))
  let enabled = includes.length === 0 || includes.some((filter) => matches(filter, packageRoot))
  if (filters.some((filter) => filter.startsWith("!") && matches(filter.slice(1), packageRoot))) enabled = false
  if (filters.some((filter) => filter.startsWith("+") && exact(filter.slice(1), packageRoot))) enabled = true
  if (filters.some((filter) => filter.startsWith("-") && exact(filter.slice(1), packageRoot))) enabled = false
  return enabled
}

const PackageManifest = Schema.Struct({ name: Schema.String, version: Schema.String, pi: Schema.Struct({ extensions: Schema.Array(Schema.String) }) })
const decodeManifest = Schema.decodeUnknown(Schema.parseJson(PackageManifest))

export const makePiCompanion = (paths: HarnessConnectionPaths, desiredSource: string = PI_COMPANION_PACKAGE_SOURCE): HarnessCompanionPackage => {
  const agentDir = dirname(paths.piSettings)
  const readSettings = readPiSettings(paths.piSettings)
  const find = (settings: PiSettings, source: PiPackageSource, byIdentity = false) =>
    pipe(Option.getOrElse(settings.packages, () => []),
      Array.map((entry, index) => ({ index, entry, source: piPackageSource(entry) })),
      Array.findFirst((candidate) => sameSource(candidate.source, source, paths.piSettings)
        || byIdentity && isCompanionNpm(candidate.source) && isCompanionNpm(source)),
    )
  const rootFor = (source: PiPackageSource) => source.startsWith("npm:")
    ? resolve(agentDir, "npm/node_modules", PI_COMPANION_PACKAGE_IDENTITY)
    : localPath(source, paths.piSettings)
  const command = (executable: string, action: "install" | "remove", source: PiPackageSource) => Effect.scoped(Effect.gen(function* () {
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
  const inspect = (source: PiPackageSource) => Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const root = rootFor(source)
    if (!(yield* fs.exists(resolve(root, "package.json")))) return false
    const manifest = yield* fs.readFileString(resolve(root, "package.json")).pipe(Effect.flatMap(decodeManifest))
    const configuredVersion = source.startsWith(`npm:${PI_COMPANION_PACKAGE_IDENTITY}@`) ? source.slice(`npm:${PI_COMPANION_PACKAGE_IDENTITY}@`.length) : "*"
    if (manifest.name !== PI_COMPANION_PACKAGE_IDENTITY
      || source.startsWith("npm:") && !satisfies(manifest.version, configuredVersion)
      || !manifest.pi.extensions.includes(`./${PI_COMPANION_EXTENSION_PATH}`)) {
      return yield* new PiPackageError({ message: `Unsupported Magnitude for Pi package at ${root}; update this package explicitly before connecting.` })
    }
    if (!(yield* fs.exists(resolve(root, PI_COMPANION_EXTENSION_PATH)))) return false
    const { metadata } = yield* verifyPluginContent(root).pipe(Effect.mapError(() => new PiPackageError({ message: `Magnitude for Pi contents at ${root} do not match its build metadata; reinstall the package.` })))
    if (metadata.rpcVersion !== MAGNITUDE_RPC_VERSION) {
      return yield* new PiPackageError({ message: `Magnitude for Pi at ${root} targets RPC ${metadata.rpcVersion}; this CLI requires RPC ${MAGNITUDE_RPC_VERSION}. Update it explicitly before connecting.` })
    }
    return yield* fs.exists(resolve(root, PI_COMPANION_EXTENSION_PATH))
  })
  const verifyHost = (executable: string) => Command.make(executable, "--version").pipe(Command.string, Effect.timeout("10 seconds"), Effect.flatMap((version) =>
    satisfies(version.trim(), ">=0.83.0") ? Effect.void
      : Effect.fail(new PiPackageError({ message: `Magnitude for Pi requires Pi 0.83.0 or newer; found ${version.trim()}.` }))))

  // Native package operations can fail after mutating settings or disk. Register the
  // inverse first, and never overwrite a changed package entry during recovery.
  const mutate = (executable: string, action: "install" | "remove", source: PiPackageSource) => Effect.gen(function* () {
    const tx = yield* ConnectionTransaction
    const beforeDocument = yield* readSettings
    const before = find(beforeDocument.settings, source)
    yield* tx.compensate(`Pi ${action} ${source}`, Effect.gen(function* () {
      const currentDocument = yield* readSettings
      const current = find(currentDocument.settings, source, true)
      if (Option.isSome(current)) {
        if (!sameSource(current.value.source, source, paths.piSettings)) {
          return yield* new PiPackageError({ message: `Preserved changed Pi package source ${current.value.source}; manual recovery needed for ${source}.` })
        }
        if (typeof current.value.entry !== "string" && !Option.containsWith(isDeepStrictEqual)(
          Option.map(before, ({ entry }) => entry), current.value.entry,
        )) {
          return yield* new PiPackageError({ message: `Preserved concurrently edited Pi package ${source}; manual recovery needed.` })
        }
      }
      yield* command(executable, action === "install" ? "remove" : "install", source)
      const afterDocument = yield* readSettings
      const packages = Option.getOrElse(afterDocument.settings.packages, () => [])
        .filter((entry) => !sameSource(piPackageSource(entry), source, paths.piSettings))
      if (Option.isSome(before)) packages.splice(Math.min(before.value.index, packages.length), 0, before.value.entry)
      // This recovery write must not register new compensation.
      const restored = yield* replacePiPackages(afterDocument,
        packages.length === 0 && Option.isNone(beforeDocument.settings.packages) ? Option.none() : Option.some(packages))
      const restoredDocument = yield* decodePiSettings(restored, paths.piSettings)
      yield* writeFileAtomic(paths.piSettings, isDeepStrictEqual(restoredDocument.settings, beforeDocument.settings)
        ? beforeDocument.text : restored)
    }))
    yield* command(executable, action, source)
  })

  return {
    description: { name: PI_COMPANION_PACKAGE_NAME, source: desiredSource, securityNotice: "Pi extensions execute with your user permissions." },
    activationInstructions: Option.some("Restart existing Pi sessions or run /reload to activate the extension."),
    reconcile: ({ installation, previous }) => Effect.gen(function* () {
      const desired = yield* Schema.decodeUnknown(PiPackageSourceSchema)(desiredSource)
      let document = yield* readSettings
      yield* verifyHost(installation.executable)
      let current = find(document.settings, desired, true).pipe(
        Option.orElse(() => Option.flatMap(previous, (state) => find(document.settings, state.source))),
      )
      let owned = Option.isSome(previous) && previous.value.ownership === "magnitude"
        && (Option.isNone(current) || sameSource(current.value.source, previous.value.source, paths.piSettings))
      let installed = false
      if (owned && Option.isSome(current) && !sameSource(current.value.source, desired, paths.piSettings)) {
        yield* mutate(installation.executable, "remove", current.value.source)
        current = Option.none()
      }
      const configuredSource = Option.match(current, { onNone: () => desired, onSome: ({ source }) => source })
      const source = configuredSource.startsWith("npm:") ? configuredSource : localPath(configuredSource, paths.piSettings)
      if (Option.isNone(current) || !(yield* inspect(source))) {
        if (Option.isNone(current)) owned = true
        yield* mutate(installation.executable, "install", source)
        if (!(yield* inspect(source))) return yield* new PiPackageError({ message: `Pi did not install a usable Magnitude extension at ${rootFor(source)}.` })
        installed = true
        document = yield* readSettings
        current = find(document.settings, source)
      }
      if (Option.isNone(current)) return yield* new PiPackageError({ message: `Pi did not register ${source}.` })
      const { entry, index } = current.value
      const enabled = piPackageExtensionEnabled(entry, rootFor(source))
      let enablement = Option.isSome(previous) && previous.value.ownership === "pre-existing"
        && sameSource(previous.value.source, source, paths.piSettings) ? previous.value.enablement : Option.none()
      if (!enabled) {
        const before = piPackageFilters(entry)
        const filters = Option.getOrElse(before, () => [])
        const after = [...filters.filter((filter) => !((filter.startsWith("+") || filter.startsWith("-")) && exact(filter.slice(1), rootFor(source)))), `+${PI_COMPANION_EXTENSION_PATH}`]
        enablement = Option.some({ before, after })
        yield* writeIfChanged(paths.piSettings, document.text, updateJsonc(document.text, [[["packages", index, "extensions"], after]]))
      }
      const state: HarnessCompanionState = owned
        ? { identity: PI_COMPANION_PACKAGE_IDENTITY, source, ownership: "magnitude" }
        : { identity: PI_COMPANION_PACKAGE_IDENTITY, source, ownership: "pre-existing", enablement }
      return { state, status: installed ? "installed" : enabled ? "already-installed" : "enabled" }
    }),
    disconnect: ({ installation, state }) => Effect.gen(function* () {
      const document = yield* readSettings
      const current = find(document.settings, state.source)
      if (Option.isNone(current)) return
      if (state.ownership === "magnitude") { yield* verifyHost(installation.executable); yield* mutate(installation.executable, "remove", state.source); return }
      if (Option.isNone(state.enablement)) return
      const { before, after } = state.enablement.value
      if (!Option.containsWith(isDeepStrictEqual)(piPackageFilters(current.value.entry), after)) return
      yield* writeIfChanged(paths.piSettings, document.text, updateJsonc(document.text, [[["packages", current.value.index, "extensions"], Option.getOrUndefined(before)]]))
    }),
  }
}

const readCommandOutput = <E, R>(stream: Stream.Stream<Uint8Array, E, R>) => stream.pipe(Stream.decodeText(), Stream.runFold("", (text, chunk) => (text + chunk).slice(-8_192)))
