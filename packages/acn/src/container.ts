import { isDockerInstallation } from "@magnitudedev/release/linux-package"
import type { NetworkAccessConfig } from "@magnitudedev/storage"
import { Effect, Option, Schema } from "effect"
import { dirname, join } from "node:path"

/**
 * A Docker installation, which its image marks in the installed resources like every Linux payload.
 * Its environment may set the network access key, which other containers and the host then use, and
 * the host names people open the browser app by (`MAGNITUDE_ALLOWED_HOSTS`, comma-separated), since
 * the container's own Settings can't change network access.
 */
export interface ContainerInstallation {
  readonly networkKey: Option.Option<string>
  readonly allowedHosts: ReadonlyArray<string>
}

/** A key other devices send as a bearer token: printable, without spaces, and long enough not to guess. */
const NetworkKey = Schema.String.pipe(Schema.pattern(/^[\x21-\x7E]{16,512}$/))

const resourcesDirectory = () => dirname(process.env.MAGNITUDE_NATIVE_HOST ?? join(dirname(process.execPath), "desktop-host.node"))

/** Whether this service runs from a Docker image's installed resources. */
export const isContainerInstallation = (resources: string = resourcesDirectory()) => isDockerInstallation(resources)


export const readContainerInstallation = (environment: Readonly<Record<string, string | undefined>> = process.env, resources: string = resourcesDirectory()) => Effect.gen(function* () {
  if (!(yield* isDockerInstallation(resources))) return Option.none<ContainerInstallation>()
  const supplied = Option.fromNullable(environment.MAGNITUDE_NETWORK_KEY?.trim()).pipe(Option.filter(key => key.length > 0))
  const networkKey = Option.filter(supplied, Schema.is(NetworkKey))
  if (Option.isSome(supplied) && Option.isNone(networkKey)) {
    yield* Effect.logWarning("Ignoring MAGNITUDE_NETWORK_KEY: it must be 16 to 512 printable characters without spaces. Magnitude uses the saved or a generated key instead.")
  }
  const allowedHosts = (environment.MAGNITUDE_ALLOWED_HOSTS ?? "").split(",").map(host => host.trim().toLowerCase()).filter(host => host.length > 0)
  return Option.some<ContainerInstallation>({ networkKey, allowedHosts })
})

/**
 * The network settings a container serves with. Published ports reach nothing on loopback, so network
 * access is on, on all interfaces, with the key required, whatever the saved `network` block says: these
 * are facts of running in a container, applied here and never saved, so a native installation's data
 * directory mounted as the volume keeps its own network settings. From the saved block the container
 * takes only its key and allowed host names. `MAGNITUDE_NETWORK_KEY`, when set, is the key, and
 * `MAGNITUDE_ALLOWED_HOSTS` adds to the allowed host names.
 */
export const containerNetworkConfig = (container: Option.Option<ContainerInstallation>, saved: Option.Option<NetworkAccessConfig>): Option.Option<NetworkAccessConfig> =>
  Option.match(container, {
    onNone: () => saved,
    onSome: ({ networkKey, allowedHosts }) => {
      const savedHosts = Option.match(saved, { onNone: () => [], onSome: block => block.allowedHosts })
      const apiKey = Option.orElse(networkKey, () => Option.flatMap(saved, block => Option.fromNullable(block.apiKey)))
      const config: NetworkAccessConfig = { enabled: true, requireApiKey: true, allowedHosts: [...new Set([...savedHosts, ...allowedHosts])] }
      return Option.some(Option.match(apiKey, { onNone: () => config, onSome: apiKey => ({ ...config, apiKey }) }))
    },
  })
