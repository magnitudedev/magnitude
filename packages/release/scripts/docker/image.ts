import { Data, Effect } from "effect"
import { isValidVersion } from "../../src/client-update/release-channels"
import { publishTag } from "../release-channel"

export const DOCKER_REPOSITORY = "magnitudedev/magnitude"

export type DockerArchitecture = "amd64" | "arm64"

export class DockerImageInputInvalid extends Data.TaggedError("DockerImageInputInvalid")<{
  readonly message: string
}> {}

export const parseDockerArchitecture = (value: string): Effect.Effect<DockerArchitecture, DockerImageInputInvalid> =>
  value === "amd64" || value === "arm64"
    ? Effect.succeed(value)
    : Effect.fail(new DockerImageInputInvalid({ message: `Architecture must be amd64 or arm64, not ${value}` }))

export const releaseHostOf = (architecture: DockerArchitecture) =>
  architecture === "amd64" ? "linux-x64-gnu" as const : "linux-arm64-gnu" as const

/** The release manifest's id for a host's Debian package. */
export const debArtifactId = (architecture: DockerArchitecture) => `desktop-${releaseHostOf(architecture)}-deb`

/**
 * Every release is tagged with its version. A stable release also moves `latest`; a prerelease moves
 * the tag of its channel (`beta`, `alpha`), the same name npm publishes it under.
 */
export const imageTags = (repository: string, version: string): Effect.Effect<readonly string[], DockerImageInputInvalid> =>
  isValidVersion(version) && /^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/.test(version)
    ? Effect.succeed([`${repository}:${version}`, `${repository}:${publishTag(version)}`])
    : Effect.fail(new DockerImageInputInvalid({ message: `${version} is not a release version that can be an image tag` }))
