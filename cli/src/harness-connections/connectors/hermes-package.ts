import * as Command from "@effect/platform/Command"
import * as FileSystem from "@effect/platform/FileSystem"
import { MAGNITUDE_RPC_VERSION } from "@magnitudedev/sdk"
import { verifyHermesPluginContent } from "@magnitudedev/release/hermes-plugin-content"
import releasePlan from "@magnitudedev/release/plan"
import { Effect, Option, Schema, Stream } from "effect"
import { dirname, resolve } from "node:path"
import { parse } from "yaml"
import type { HarnessCompanionPackage } from "../contract"
import type { HarnessConnectionPaths } from "../paths"
import { readOr, updateYaml, valueAt, writeIfChanged } from "../shared"
import { ConnectionTransaction } from "../transaction"
import {
  HermesPackageRevisionSchema, HermesPackageSourceSchema,
  HermesPackageSelectionSchema, type HermesPackageSelection,
} from "./hermes-package-state"

const selectedPackage = releasePlan.plugins.map(({ artifact }) => artifact).find(artifact => artifact.host === "hermes")
export const HERMES_COMPANION_SELECTION = Schema.decodeUnknownSync(HermesPackageSelectionSchema)({
  source: selectedPackage?.repository,
  revision: selectedPackage?.revision,
  contentFingerprint: selectedPackage?.contentFingerprint,
})

export class HermesPackageError extends Schema.TaggedError<HermesPackageError>()("HermesPackageError", {
  message: Schema.String,
}) {}

const NativeInstallRecord = Schema.Struct({
  source: HermesPackageSourceSchema,
  revision: HermesPackageRevisionSchema,
  pinned: Schema.Boolean,
})
const NativeInstallMetadata = Schema.Struct({ magnitude: NativeInstallRecord })
const PluginLists = Schema.Struct({
  enabled: Schema.optionalWith(Schema.Array(Schema.String), { as: "Option", exact: true }),
  disabled: Schema.optionalWith(Schema.Array(Schema.String), { as: "Option", exact: true }),
})
const identity = "@magnitudedev/hermes-companion" as const
const pluginName = "magnitude"

/** Native Git installation, with receipts independent of Hermes's model configuration. */
export const makeHermesCompanion = (
  paths: Pick<HarnessConnectionPaths, "hermes">,
  desired: HermesPackageSelection = HERMES_COMPANION_SELECTION,
): HarnessCompanionPackage => {
  const home = dirname(paths.hermes)
  const root = resolve(home, "plugins", pluginName)
  const readLists = Effect.gen(function* () {
    const text = yield* readOr(paths.hermes, "{}\n")
    const document = yield* Effect.try(() => parse(text))
    const lists = yield* Schema.decodeUnknown(PluginLists)(valueAt(document, ["plugins"]) ?? {})
    return {
      text,
      enabled: Option.getOrElse(lists.enabled, (): readonly string[] => []),
      disabled: Option.getOrElse(lists.disabled, (): readonly string[] => []),
    }
  })
  const setMembership = (enabled: boolean, disabled: boolean) => Effect.gen(function* () {
    const current = yield* readLists
    const membership = (values: readonly string[], present: boolean) => present
      ? [...values.filter(value => value !== pluginName), pluginName]
      : values.filter(value => value !== pluginName)
    yield* writeIfChanged(paths.hermes, current.text, updateYaml(current.text, [
      [["plugins", "enabled"], membership(current.enabled, enabled)],
      [["plugins", "disabled"], membership(current.disabled, disabled)],
    ]))
  })
  const command = (executable: string, args: readonly string[]) => Effect.scoped(Effect.gen(function* () {
    const child = yield* Command.make(executable, "plugins", ...args).pipe(
      Command.env({ HERMES_HOME: home }), Command.start,
    )
    const text = <E, R>(stream: Stream.Stream<Uint8Array, E, R>) => stream.pipe(
      Stream.decodeText(), Stream.runFold("", (text, chunk) => (text + chunk).slice(-8_192)),
    )
    const [code, stdout, stderr] = yield* Effect.all([
      child.exitCode, text(child.stdout), text(child.stderr),
    ], { concurrency: "unbounded" })
    if (Number(code) !== 0) return yield* new HermesPackageError({
      message: `Hermes plugin ${args[0]} failed (${code}): ${stderr || stdout}`,
    })
  })).pipe(Effect.timeout("2 minutes"))

  const inspect = Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    if (!(yield* fs.exists(root))) return Option.none()
    const inspected = yield* verifyHermesPluginContent(root).pipe(Effect.mapError(() => new HermesPackageError({
      message: `Preserved the Magnitude plugin at ${root}: its contents do not match its build metadata. Reinstall it explicitly before connecting.`,
    })))
    if (inspected.metadata.rpcVersion !== MAGNITUDE_RPC_VERSION) return yield* new HermesPackageError({
      message: `Magnitude for Hermes uses RPC ${inspected.metadata.rpcVersion}; this CLI requires RPC ${MAGNITUDE_RPC_VERSION}. Update the plugin explicitly before connecting.`,
    })
    return Option.some(inspected.metadata)
  })

  // Content hashes establish identity; native provenance and the complete file
  // set additionally establish whether removing this directory is safe.
  const verifyOwned = (selection: HermesPackageSelection) => Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const inspected = yield* verifyHermesPluginContent(root)
    const metadata = yield* fs.readFileString(resolve(home, "plugins/.install-metadata.json")).pipe(
      Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(NativeInstallMetadata))),
    )
    const record = metadata[pluginName]
    if (record === undefined || !record.pinned || record.source !== selection.source
      || record.revision !== selection.revision
      || inspected.metadata.contentFingerprint !== selection.contentFingerprint) {
      return yield* new HermesPackageError({ message: `Preserved changed Hermes plugin installation at ${root}; reconnect or remove it explicitly.` })
    }
    const expected = new Set([...Object.keys(inspected.metadata.files), "dist/magnitude-plugin.json"])
    const actualRoot = yield* fs.realPath(root)
    if (actualRoot !== resolve(yield* fs.realPath(dirname(root)), pluginName)) {
      return yield* new HermesPackageError({ message: `Preserved symlinked Hermes plugin at ${root}.` })
    }
    const checkDirectory = (relative: string): Effect.Effect<void, unknown, FileSystem.FileSystem> => Effect.gen(function* () {
      for (const name of yield* fs.readDirectory(resolve(root, relative))) {
        const path = relative === "" ? name : `${relative}/${name}`
        if ((yield* fs.realPath(resolve(root, path))) !== resolve(actualRoot, path)) {
          return yield* new HermesPackageError({ message: `Preserved symlinked Hermes plugin file ${path}.` })
        }
        if (path === ".git") continue
        const stat = yield* fs.stat(resolve(root, path))
        if (stat.type === "Directory" && (name === "__pycache__" || [...expected].some(file => file.startsWith(`${path}/`)))) {
          yield* checkDirectory(path)
        } else if (stat.type !== "File" || (!expected.has(path) && !(relative.endsWith("__pycache__") && name.endsWith(".pyc")))) {
          return yield* new HermesPackageError({ message: `Preserved extra Hermes plugin file ${path}; move local files out before removing or updating the plugin.` })
        }
      }
    })
    yield* checkDirectory("")
  })

  const install = (executable: string, selection: HermesPackageSelection) => Effect.gen(function* () {
    const tx = yield* ConnectionTransaction
    yield* tx.compensate("Remove newly installed Hermes companion", Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      if (!(yield* fs.exists(root))) return
      yield* verifyOwned(selection)
      yield* command(executable, ["remove", pluginName])
    }))
    // No --force: Hermes's native security scanner remains authoritative.
    // Magnitude changes only its own enablement membership after installation.
    yield* command(executable, ["install", selection.source, "--ref", selection.revision, "--no-enable"])
    yield* verifyOwned(selection)
  })
  const remove = (executable: string, selection: HermesPackageSelection) => Effect.gen(function* () {
    yield* verifyOwned(selection)
    const tx = yield* ConnectionTransaction
    yield* tx.compensate("Restore removed Hermes companion", Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      if (yield* fs.exists(root)) {
        yield* verifyOwned(selection)
        return
      }
      yield* command(executable, ["install", selection.source, "--ref", selection.revision, "--no-enable"])
      yield* verifyOwned(selection)
    }))
    yield* command(executable, ["remove", pluginName])
  })

  return {
    description: {
      name: "Magnitude for Hermes", source: `${desired.source} --ref ${desired.revision}`,
      securityNotice: "Hermes plugins execute with your user permissions.",
    },
    activationInstructions: Option.some("Restart Hermes to activate the companion. In Hermes Desktop, enable Magnitude in Settings → Plugins."),
    reconcile: ({ installation, previous: receipt }) => Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const previous = Option.filter(receipt, state => state.identity === identity)
      const lists = yield* readLists
      const enabled = lists.enabled.includes(pluginName) && !lists.disabled.includes(pluginName)
      let owned = Option.isSome(previous) && previous.value.ownership === "magnitude"
      let installed = false
      if (owned && Option.isSome(previous) && previous.value.ownership === "magnitude"
        && (yield* fs.exists(root))) {
        yield* verifyOwned(previous.value)
        if (previous.value.source !== desired.source || previous.value.revision !== desired.revision) {
          yield* remove(installation.executable, previous.value)
        }
      }
      if (!(yield* fs.exists(root))) {
        yield* install(installation.executable, desired)
        owned = true
        installed = true
      }
      yield* inspect
      const enablement = Option.isSome(previous) && Option.isSome(previous.value.enablement)
        ? previous.value.enablement
        : enabled ? Option.none() : Option.some({ enabled: lists.enabled.includes(pluginName), disabled: lists.disabled.includes(pluginName) })
      if (!enabled) yield* setMembership(true, false)
      return {
        state: owned
          ? { identity, ...desired, ownership: "magnitude" as const, enablement }
          : { identity, source: HermesPackageSourceSchema.make(root), ownership: "pre-existing" as const, enablement },
        status: installed ? "installed" : enabled ? "already-installed" : "enabled",
      }
    }),
    disconnect: ({ installation, state }) => Effect.gen(function* () {
      if (state.identity !== identity) return yield* new HermesPackageError({ message: "The companion receipt does not belong to Hermes." })
      const fs = yield* FileSystem.FileSystem
      if (state.ownership === "magnitude" && (yield* fs.exists(root))) yield* remove(installation.executable, state)
      if (Option.isNone(state.enablement)) return
      const lists = yield* readLists
      if (lists.enabled.includes(pluginName) && !lists.disabled.includes(pluginName)) {
        yield* setMembership(state.enablement.value.enabled, state.enablement.value.disabled)
      }
    }),
  }
}
