import { FetchHttpClient } from "@effect/platform"
import { Effect, Layer, Option } from "effect"
import { join } from "node:path"
import { MagnitudeClient, type NetworkAccessSettings } from "@magnitudedev/sdk"
import { makeFirstPartyConnection } from "@magnitudedev/client-common"

/** One request to the service on this machine's loopback port, without starting or observing an owner. */
export const withLocalService = <A, E>(origin: string, use: (client: MagnitudeClient) => Effect.Effect<A, E>) => Effect.scoped(
  makeFirstPartyConnection(MagnitudeClient.layer({ origin, autoStart: false }).pipe(Layer.provide(FetchHttpClient.layer))).pipe(
    Effect.flatMap(connection => use(connection.client)), Effect.timeout("5 seconds")))

export interface ServerReach {
  readonly endpoint: string
  readonly network: Option.Option<Pick<NetworkAccessSettings, "enabled" | "interfaces" | "port" | "pending">>
  readonly configPath: string
  /** Running as the system service: stopping it means removing the server, not Ctrl+C. */
  readonly service: boolean
}

const host = (address: string) => address.includes(":") ? `[${address}]` : address

/**
 * How to reach a ready server, shared by `serve` and `status`. It never includes the network
 * access key: service output goes to logs other accounts can read.
 */
export const renderServerReach = (reach: ServerReach): readonly string[] => {
  const lines = [`On this computer:   ${reach.endpoint}`]
  const network = Option.getOrUndefined(reach.network)
  const addresses = network?.enabled ? network.interfaces.filter(entry => entry.kind !== "virtual") : []
  if (network?.enabled && addresses.length > 0) {
    lines.push("From other devices: open one of these in a browser; it asks for the network access key.")
    for (const entry of addresses) lines.push(`  http://${host(entry.address)}:${network.port}${entry.kind === "tailscale" ? "  (Tailscale)" : ""}`)
    lines.push(`Apps and agents:    http://${host(addresses[0]!.address)}:${network.port}/inference/v1`)
  } else if (network?.enabled) {
    lines.push(`From other devices: http://<this machine>:${network.port}; the browser app asks for the network access key.`)
  } else {
    lines.push(
      "Only this computer can reach it. To use it from other devices, turn on network access in",
      `Settings → Network access, or set "network": { "enabled": true } in ${reach.configPath} and restart.`,
      "Or reach it through an SSH tunnel: ssh -N -L 10100:127.0.0.1:10100 <you>@<this machine>",
    )
  }
  if (network?.pending) lines.push("A network access change applies when Magnitude restarts.")
  return lines
}

/** Reads network access from the service itself; reach falls back to this computer only if it can't. */
export const readServerReach = (options: { readonly endpoint: string; readonly dataDirectory: string; readonly service: boolean }) =>
  withLocalService(options.endpoint, client => client.configuration.getNetworkAccess({})).pipe(
    Effect.option,
    Effect.map((network): ServerReach => ({ endpoint: options.endpoint, network, configPath: join(options.dataDirectory, "config.json"), service: options.service })))

export const renderServeReady = (reach: ServerReach) => [
  "Magnitude is ready.",
  ...renderServerReach(reach),
  reach.service ? "Stop it with `magnitude server remove`." : "Press Ctrl+C to stop.",
  "",
].join("\n")
