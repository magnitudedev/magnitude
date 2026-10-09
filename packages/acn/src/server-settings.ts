import { FileSystem } from "@effect/platform"
import { Context, Effect, Layer, Option } from "effect"
import { createRequire } from "node:module"
import { isIP } from "node:net"
import { homedir, networkInterfaces } from "node:os"
import { dirname, isAbsolute, join, normalize, parse } from "node:path"
import {
  Configuration,
  ServerSettingsFailed,
  type DirectoryListing,
  type ModelStorageSettings,
  type NetworkAccessChange,
  type NetworkAccessSettings,
  type NetworkInterfaceAddress,
  type ServerMachine,
} from "@magnitudedev/acn-protocol"
import {
  GlobalStorage,
  LOOPBACK_ONLY,
  MagnitudeConfigSchema,
  generateApiKey,
  makeConfigStorage,
  makeGlobalStorage,
  readStructuredFile,
  resolveModelStoreLocation,
  resolveNetworkAccess,
  type NetworkAccess,
  type NetworkAccessConfig,
} from "@magnitudedev/storage"
import { AcnChanges } from "./changes"
import { readMachineIdentity } from "./machine-identity"

/** Facts about this ACN process fixed at startup: where it keeps data and what it serves. */
export interface AcnHostApi {
  readonly dataDir: string
  readonly port: number
  readonly activeNetwork: NetworkAccess
}
export class AcnHost extends Context.Tag("AcnHost")<AcnHost, AcnHostApi>() {}

export interface ServerSettingsApi {
  readonly modelStorage: Effect.Effect<ModelStorageSettings, ServerSettingsFailed>
  readonly setModelStorage: (path: Option.Option<string>) => Effect.Effect<void, ServerSettingsFailed>
  readonly browseDirectories: (path: Option.Option<string>) => Effect.Effect<DirectoryListing, ServerSettingsFailed>
  readonly networkAccess: Effect.Effect<NetworkAccessSettings, ServerSettingsFailed>
  readonly setNetworkAccess: (change: NetworkAccessChange) => Effect.Effect<void, ServerSettingsFailed>
  readonly regenerateNetworkApiKey: Effect.Effect<void, ServerSettingsFailed>
  readonly machine: Effect.Effect<ServerMachine>
}
export class ServerSettings extends Context.Tag("ServerSettings")<ServerSettings, ServerSettingsApi>() {}

/** What the service actually enforces; two settings that resolve the same way need no restart. */
export const networkAccessEquals = (a: NetworkAccess, b: NetworkAccess): boolean =>
  a.enabled === b.enabled && a.bind === b.bind && a.requireApiKey === b.requireApiKey
  && Option.getOrNull(a.apiKey) === Option.getOrNull(b.apiKey)
  && a.allowedHosts.length === b.allowedHosts.length && a.allowedHosts.every((host, index) => host === b.allowedHosts[index])

const isTailscaleAddress = (address: string) => {
  const [first, second] = address.split(".").map(Number)
  return first === 100 && second !== undefined && second >= 64 && second <= 127
}
const VIRTUAL_INTERFACE = /^(bridge|vmnet|vboxnet|docker|veth|br-|virbr|utun|tun|tap|wg|ppp|llw|awdl|anpi|ap\d|vEthernet|VirtualBox|VMware)/i
const KIND_ORDER: Record<NetworkInterfaceAddress["kind"], number> = { lan: 0, tailscale: 1, virtual: 2 }

/** IPv4 addresses other devices could use: physical networks first, then Tailscale, then virtual adapters. */
export const listNetworkInterfaces = (interfaces: ReturnType<typeof networkInterfaces> = networkInterfaces()): ReadonlyArray<NetworkInterfaceAddress> =>
  Object.entries(interfaces).flatMap(([name, entries]) => (entries ?? [])
    .filter(entry => entry.family === "IPv4" && !entry.internal && !entry.address.startsWith("169.254."))
    .map(entry => ({ name, address: entry.address, kind: isTailscaleAddress(entry.address) ? "tailscale" as const : VIRTUAL_INTERFACE.test(name) ? "virtual" as const : "lan" as const })))
    .sort((a, b) => KIND_ORDER[a.kind] - KIND_ORDER[b.kind])

const nativeHostPath = () => process.env.MAGNITUDE_NATIVE_HOST ?? join(dirname(process.execPath), "desktop-host.node")

export const ServerSettingsLive = Layer.effect(ServerSettings, Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const host = yield* AcnHost
  const changes = yield* AcnChanges
  const storage = makeGlobalStorage({ root: host.dataDir })
  const config = yield* makeConfigStorage().pipe(Effect.provideService(GlobalStorage, storage))
  const writes = yield* Effect.makeSemaphore(1)
  const readModelsDirectory = readStructuredFile(storage.paths.configFile, MagnitudeConfigSchema.pick("modelsDirectory")).pipe(
    Effect.provideService(FileSystem.FileSystem, fs),
    Effect.flatMap(result => result._tag === "Invalid" ? Effect.fail(result.error) : Effect.succeed(result._tag === "Missing" ? Option.none<string>() : result.value.modelsDirectory)),
  )
  const location = (configured: Option.Option<string>) => resolveModelStoreLocation(host.dataDir, configured).pipe(Effect.provideService(FileSystem.FileSystem, fs))
  // The engine received its store at spawn, from the configuration as it stood when this service started.
  const active = yield* readModelsDirectory.pipe(Effect.orElseSucceed(() => Option.none<string>()), Effect.flatMap(location))
  const defaultPath = (yield* location(Option.none())).path

  const readNetwork = readStructuredFile(storage.paths.configFile, MagnitudeConfigSchema.pick("network")).pipe(
    Effect.provideService(FileSystem.FileSystem, fs),
    Effect.flatMap(result => result._tag === "Invalid" ? Effect.fail(result.error) : Effect.succeed(result._tag === "Missing" ? Option.none<NetworkAccessConfig>() : result.value.network)),
  )
  const writeNetwork = (f: (current: Option.Option<NetworkAccessConfig>) => NetworkAccessConfig) =>
    config.update(current => ({ ...current, network: Option.some(f(current.network)) })).pipe(
      Effect.asVoid,
      Effect.mapError(() => new ServerSettingsFailed({ message: "Network access could not be saved. Check access to the Magnitude configuration and try again." })),
    )
  const networkBase = (current: Option.Option<NetworkAccessConfig>): NetworkAccessConfig => Option.getOrElse(current, () => ({ enabled: false, requireApiKey: true, allowedHosts: [] }))
  const poke = (operation: string) => changes.publish({ operation })

  const machine = yield* Effect.cached(readMachineIdentity(() => createRequire(import.meta.url)(nativeHostPath())).pipe(
    Effect.map((identity): ServerMachine => ({ platform: process.platform, identity })),
  ))

  return ServerSettings.of({
    modelStorage: readModelsDirectory.pipe(
      Effect.flatMap(location),
      Effect.map((current): ModelStorageSettings => ({ active: active.path, path: current.path, source: current.source, defaultPath, warning: current.warning })),
      Effect.mapError(() => new ServerSettingsFailed({ message: "The saved model storage location could not be read. Using the default folder." })),
    ),
    setModelStorage: path => Effect.gen(function* () {
      const trimmed = Option.map(path, value => value.trim())
      if (Option.isSome(trimmed) && (trimmed.value.length === 0 || !isAbsolute(trimmed.value))) {
        return yield* new ServerSettingsFailed({ message: "Choose a full folder path for model storage." })
      }
      yield* writes.withPermits(1)(config.update(current => ({ ...current, modelsDirectory: trimmed }))).pipe(
        Effect.mapError(() => new ServerSettingsFailed({ message: "The model storage location could not be saved. Check access to the Magnitude configuration and try again." })),
      )
      yield* poke(Configuration.getModelStorage._tag)
    }),
    browseDirectories: path => Effect.gen(function* () {
      const target = normalize(Option.getOrElse(path, () => homedir()))
      if (!isAbsolute(target)) return yield* new ServerSettingsFailed({ message: "Choose a full folder path." })
      const names = yield* fs.readDirectory(target).pipe(Effect.mapError(() => new ServerSettingsFailed({ message: "This folder can't be opened. Check that it exists and that Magnitude can read it." })))
      const directories = yield* Effect.forEach(names.filter(name => !name.startsWith(".")), name => {
        const child = join(target, name)
        return fs.stat(child).pipe(Effect.map(info => info.type === "Directory" ? Option.some({ name, path: child }) : Option.none()), Effect.orElseSucceed(() => Option.none()))
      }, { concurrency: 16 })
      const root = parse(target).root
      return {
        path: target,
        parent: target === root ? Option.none() : Option.some(dirname(target)),
        directories: directories.flatMap(entry => Option.isSome(entry) ? [entry.value] : []).sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: "base" })),
      }
    }),
    networkAccess: readNetwork.pipe(
      Effect.map((saved): NetworkAccessSettings => {
        const resolved = Option.isNone(saved) ? LOOPBACK_ONLY : resolveNetworkAccess(saved)
        const current = networkBase(saved)
        return {
          enabled: current.enabled,
          bind: Option.fromNullable(current.bind),
          requireApiKey: current.requireApiKey,
          apiKey: Option.fromNullable(current.apiKey),
          interfaces: listNetworkInterfaces(),
          port: host.port,
          pending: !networkAccessEquals(resolved, host.activeNetwork),
          warning: resolved.warning,
        }
      }),
      Effect.mapError(() => new ServerSettingsFailed({ message: "The saved network access settings could not be read. Network access is off." })),
    ),
    setNetworkAccess: change => Effect.gen(function* () {
      if (Option.isSome(change.bind) && change.bind.value._tag === "Address" && isIP(change.bind.value.address) === 0) {
        return yield* new ServerSettingsFailed({ message: "Choose an address from the list, or all interfaces." })
      }
      yield* writes.withPermits(1)(writeNetwork(current => {
        const existing = networkBase(current)
        const enabled = Option.getOrElse(change.enabled, () => existing.enabled)
        const bind = Option.match(change.bind, {
          onNone: () => existing.bind,
          onSome: value => value._tag === "Address" ? value.address : undefined,
        })
        return {
          ...existing,
          enabled,
          bind,
          requireApiKey: Option.getOrElse(change.requireApiKey, () => existing.requireApiKey),
          apiKey: existing.apiKey ?? (enabled ? generateApiKey() : undefined),
        }
      }))
      yield* poke(Configuration.getNetworkAccess._tag)
    }),
    regenerateNetworkApiKey: writes.withPermits(1)(writeNetwork(current => ({ ...networkBase(current), apiKey: generateApiKey() }))).pipe(
      Effect.zipRight(poke(Configuration.getNetworkAccess._tag)),
    ),
    machine,
  })
}))
