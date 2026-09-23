import { FileSystem } from "@effect/platform"
import { GlobalStorage, makeGlobalStorage, makeConfigStorage, MagnitudeConfigSchema, readStructuredFile, resolveNetworkAccess, generateApiKey, LOOPBACK_ONLY, type NetworkAccess, type NetworkAccessConfig } from "@magnitudedev/storage"
import { Context, Effect, Option, Schema } from "effect"
import { isIP } from "node:net"
import { networkInterfaces } from "node:os"

export type { NetworkAccess } from "@magnitudedev/storage"
export { LOOPBACK_ONLY } from "@magnitudedev/storage"

export class NetworkPreferencesFailed extends Schema.TaggedError<NetworkPreferencesFailed>()("NetworkPreferencesFailed", {
  message: Schema.String,
}) {}

export interface NetworkAccessChange {
  readonly enabled?: boolean
  readonly bind?: string | null
  readonly requireApiKey?: boolean
}

export interface NetworkPreferences {
  readonly read: Effect.Effect<{ readonly saved: Option.Option<NetworkAccessConfig>; readonly resolved: NetworkAccess }, NetworkPreferencesFailed>
  readonly update: (change: NetworkAccessChange) => Effect.Effect<void, NetworkPreferencesFailed>
  readonly regenerateApiKey: Effect.Effect<void, NetworkPreferencesFailed>
}
export const NetworkPreferences = Context.GenericTag<NetworkPreferences>("desktop/NetworkPreferences")

/** What the service actually enforces; two settings that resolve the same way need no restart. */
export const networkAccessEquals = (a: NetworkAccess, b: NetworkAccess): boolean =>
  a.enabled === b.enabled && a.bind === b.bind && a.requireApiKey === b.requireApiKey
  && Option.getOrNull(a.apiKey) === Option.getOrNull(b.apiKey)
  && a.allowedHosts.length === b.allowedHosts.length && a.allowedHosts.every((host, index) => host === b.allowedHosts[index])

export type NetworkInterfaceKind = "lan" | "tailscale" | "virtual"
export interface NetworkInterfaceAddress {
  readonly name: string
  readonly address: string
  readonly kind: NetworkInterfaceKind
}

const isTailscaleAddress = (address: string) => {
  const [first, second] = address.split(".").map(Number)
  return first === 100 && second !== undefined && second >= 64 && second <= 127
}
const VIRTUAL_INTERFACE = /^(bridge|vmnet|vboxnet|docker|veth|br-|virbr|utun|tun|tap|wg|ppp|llw|awdl|anpi|ap\d|vEthernet|VirtualBox|VMware)/i
const KIND_ORDER: Record<NetworkInterfaceKind, number> = { lan: 0, tailscale: 1, virtual: 2 }

/** IPv4 addresses other devices could use: physical networks first, then Tailscale, then virtual adapters. */
export const listNetworkInterfaces = (interfaces: ReturnType<typeof networkInterfaces> = networkInterfaces()): ReadonlyArray<NetworkInterfaceAddress> =>
  Object.entries(interfaces).flatMap(([name, entries]) => (entries ?? [])
    .filter(entry => entry.family === "IPv4" && !entry.internal && !entry.address.startsWith("169.254."))
    .map(entry => ({ name, address: entry.address, kind: isTailscaleAddress(entry.address) ? "tailscale" as const : VIRTUAL_INTERFACE.test(name) ? "virtual" as const : "lan" as const })))
    .sort((a, b) => KIND_ORDER[a.kind] - KIND_ORDER[b.kind])

export const makeNetworkPreferences = (clientDataDirectory: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const storage = makeGlobalStorage({ root: clientDataDirectory })
  const config = yield* makeConfigStorage().pipe(Effect.provideService(GlobalStorage, storage))
  const saveFailed = () => new NetworkPreferencesFailed({ message: "Network access could not be saved. Check access to the Magnitude configuration and try again." })
  const write = (f: (current: Option.Option<NetworkAccessConfig>) => NetworkAccessConfig) => config.update(current => ({ ...current, network: Option.some(f(current.network)) })).pipe(
    Effect.asVoid, Effect.mapError(saveFailed))
  const base = (current: Option.Option<NetworkAccessConfig>): NetworkAccessConfig => Option.getOrElse(current, () => ({ enabled: false, requireApiKey: true, allowedHosts: [] }))
  return NetworkPreferences.of({
    read: readStructuredFile(storage.paths.configFile, MagnitudeConfigSchema.pick("network")).pipe(
      Effect.provideService(FileSystem.FileSystem, fs),
      Effect.flatMap(result => result._tag === "Invalid" ? Effect.fail(result.error) : Effect.succeed(result._tag === "Missing" ? Option.none<NetworkAccessConfig>() : result.value.network)),
      Effect.map(saved => ({ saved, resolved: Option.isNone(saved) ? LOOPBACK_ONLY : resolveNetworkAccess(saved) })),
      Effect.mapError(() => new NetworkPreferencesFailed({ message: "The saved network access settings could not be read. Network access is off." })),
    ),
    update: change => Effect.gen(function* () {
      if (change.bind !== undefined && change.bind !== null && isIP(change.bind) === 0) {
        return yield* new NetworkPreferencesFailed({ message: "Choose an address from the list, or all interfaces." })
      }
      yield* write(current => {
        const existing = base(current)
        const enabled = change.enabled ?? existing.enabled
        return {
          ...existing,
          enabled,
          bind: change.bind === undefined ? existing.bind : change.bind === null ? undefined : change.bind,
          requireApiKey: change.requireApiKey ?? existing.requireApiKey,
          apiKey: existing.apiKey ?? (enabled ? generateApiKey() : undefined),
        }
      })
    }),
    regenerateApiKey: write(current => ({ ...base(current), apiKey: generateApiKey() })),
  })
})
