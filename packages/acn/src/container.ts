import { FileSystem } from "@effect/platform"
import { LinuxInstallationKind } from "@magnitudedev/release/linux-package"
import type { NetworkAccessConfig } from "@magnitudedev/storage"
import { Effect, Option, Schema } from "effect"
import { dirname, join } from "node:path"

/**
 * A Docker installation, which its image marks in the installed resources like every Linux payload.
 * Its environment may set the network access key, which other containers and the host then use.
 */
export interface ContainerInstallation {
  readonly networkKey: Option.Option<string>
}

/** A key other devices send as a bearer token: printable, without spaces, and long enough not to guess. */
const NetworkKey = Schema.String.pipe(Schema.pattern(/^[\x21-\x7E]{16,512}$/))

const resourcesDirectory = () => dirname(process.env.MAGNITUDE_NATIVE_HOST ?? join(dirname(process.execPath), "desktop-host.node"))

/** Whether the installed resources are a Docker image's; an unreadable marker means they aren't. */
export const isContainerInstallation = (resources: string = resourcesDirectory()) => Effect.flatMap(FileSystem.FileSystem, fs =>
  fs.readFileString(join(resources, "update-package.json")).pipe(
    Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(Schema.Struct({ format: LinuxInstallationKind })))),
    Effect.map(marker => marker.format === "docker"), Effect.orElseSucceed(() => false)))

export const readContainerInstallation = (environment: Readonly<Record<string, string | undefined>> = process.env, resources: string = resourcesDirectory()) => Effect.gen(function* () {
  if (!(yield* isContainerInstallation(resources))) return Option.none<ContainerInstallation>()
  const supplied = Option.fromNullable(environment.MAGNITUDE_NETWORK_KEY?.trim()).pipe(Option.filter(key => key.length > 0))
  const networkKey = Option.filter(supplied, Schema.is(NetworkKey))
  if (Option.isSome(supplied) && Option.isNone(networkKey)) {
    yield* Effect.logWarning("Ignoring MAGNITUDE_NETWORK_KEY: it must be 16 to 512 printable characters without spaces. Magnitude uses the saved or a generated key instead.")
  }
  return Option.some<ContainerInstallation>({ networkKey })
})

/**
 * The network settings a container serves with. Published ports reach nothing on loopback, so without
 * a `network` block network access is on, on all interfaces, with the key required; a block the person
 * saved still decides. `MAGNITUDE_NETWORK_KEY`, when set, is the key either way.
 */
export const containerNetworkConfig = (container: Option.Option<ContainerInstallation>, saved: Option.Option<NetworkAccessConfig>): Option.Option<NetworkAccessConfig> =>
  Option.match(container, {
    onNone: () => saved,
    onSome: ({ networkKey }) => {
      const config = Option.getOrElse(saved, (): NetworkAccessConfig => ({ enabled: true, requireApiKey: true, allowedHosts: [] }))
      return Option.some(Option.match(networkKey, { onNone: () => config, onSome: apiKey => ({ ...config, apiKey }) }))
    },
  })
