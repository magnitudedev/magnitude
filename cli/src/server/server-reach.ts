import { FetchHttpClient } from "@effect/platform"
import { Config, Effect, Layer, Option } from "effect"
import { join } from "node:path"
import { MagnitudeClient, type NetworkAccessSettings } from "@magnitudedev/sdk"
import { makeFirstPartyConnection } from "@magnitudedev/client-common"

/**
 * One request to the service on this machine's loopback port, without starting or observing an owner.
 * The bound covers connecting and closing too, which cannot be interrupted, so a service that accepts
 * the connection and never answers cannot hold the caller.
 */
export const withLocalService = <A, E>(origin: string, use: (client: MagnitudeClient) => Effect.Effect<A, E>) => Effect.scoped(
  makeFirstPartyConnection(MagnitudeClient.layer({ origin, autoStart: false }).pipe(Layer.provide(FetchHttpClient.layer))).pipe(
    Effect.flatMap(connection => use(connection.client)))).pipe(Effect.disconnect, Effect.timeout("5 seconds"))

export interface ServerReach {
  readonly endpoint: string
  readonly network: Option.Option<Pick<NetworkAccessSettings, "enabled" | "bind" | "interfaces" | "port" | "pending" | "container">>
  readonly configPath: string
  /** Running as the system service: stopping it means removing the server, not Ctrl+C. */
  readonly service: boolean
  /** Running in a Kubernetes pod, where the container is reached and stopped through kubectl, not Docker. */
  readonly kubernetes: boolean
}

const host = (address: string) => address.includes(":") ? `[${address}]` : address

/** The SSH tunnel to the loopback endpoint, forwarding the same port on the other device. */
const sshTunnel = (endpoint: string) => {
  const url = new URL(endpoint)
  const port = url.port.length > 0 ? url.port : url.protocol === "https:" ? "443" : "80"
  return `ssh -N -L ${port}:${url.hostname}:${port} <you>@<this machine>`
}

/**
 * How to reach a ready server, shared by `serve` and `status`. It never includes the network
 * access key: service output goes to logs other accounts can read.
 */
export const renderServerReach = (reach: ServerReach): readonly string[] => {
  const network = Option.getOrUndefined(reach.network)
  if (network?.container) return renderContainerReach(reach, network)
  const lines = [`On this computer:   ${reach.endpoint}`]
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
      `Or reach it through an SSH tunnel: ${sshTunnel(reach.endpoint)}`,
    )
  }
  if (network?.pending) lines.push("A network access change applies when Magnitude restarts.")
  return lines
}

/**
 * A container's own addresses (172.17.0.x) are reachable only from its Docker network, so reach is
 * described from the host and the other containers, through the port each of them publishes or uses.
 * In a Kubernetes pod the same holds for the pod's address, through a Service and port forwarding.
 */
const renderContainerReach = (reach: ServerReach, network: NonNullable<Option.Option.Value<ServerReach["network"]>>): readonly string[] => {
  if (!network.enabled) return [
    `Only this ${reach.kubernetes ? "pod" : "container"} can reach it: network access is off in ${reach.configPath}. Remove that setting or turn it on, and restart the ${reach.kubernetes ? "pod" : "container"}.`,
    ...(network.pending ? ["A network access change applies when Magnitude restarts."] : []),
  ]
  return [
    ...(reach.kubernetes ? [
      `Other pods:         http://<the Service's name>:${network.port}, through a Service for the pod's port ${network.port}`,
      `Your computer:      kubectl port-forward <pod> ${network.port}, then http://localhost:${network.port}`,
    ] : [
      `On the host:        http://localhost:${network.port}, or the host port you published for the container's port ${network.port}`,
      `Other containers:   http://<this container's name>:${network.port} on a shared Docker network`,
    ]),
    `Apps and agents:    the address above with /inference/v1, sending the network access key as a Bearer token`,
    `The key is network.apiKey in ${reach.configPath}: ${reach.kubernetes ? "kubectl exec <pod> --" : "docker exec <container>"} cat ${reach.configPath}`,
    ...(network.pending ? ["A network access change applies when Magnitude restarts."] : []),
  ]
}

/** Reads network access from the service itself; reach falls back to this computer only if it can't. */
export const readServerReach = (options: { readonly endpoint: string; readonly dataDirectory: string; readonly service: boolean }) =>
  withLocalService(options.endpoint, client => client.configuration.getNetworkAccess({})).pipe(
    Effect.option,
    Effect.zip(Effect.option(Config.string("KUBERNETES_SERVICE_HOST"))),
    Effect.map(([network, kubernetes]): ServerReach => ({
      endpoint: options.endpoint, network, configPath: join(options.dataDirectory, "config.json"), service: options.service, kubernetes: Option.isSome(kubernetes),
    })))

/** How to stop the system service: systemd stops a Linux service in place; the macOS daemon is removed. */
const stopService = (platform: NodeJS.Platform) => platform === "linux"
  ? "Stop it with `sudo systemctl stop magnitude`; `magnitude server remove` removes the server."
  : "Stop it with `magnitude server remove`."

/** Deleting a pod only makes its workload start another, so a pod is stopped by scaling the workload down. */
const stopContainer = (reach: ServerReach) => reach.kubernetes
  ? "Stop it with `kubectl scale deployment <name> --replicas=0`; deleting the pod only starts another."
  : "Stop it with `docker stop <container>`."

export const renderServeReady = (reach: ServerReach, platform: NodeJS.Platform = process.platform) => [
  "Magnitude is ready.",
  ...renderServerReach(reach),
  Option.exists(reach.network, network => network.container) ? stopContainer(reach)
    : reach.service ? stopService(platform) : "Press Ctrl+C to stop.",
  "",
].join("\n")
