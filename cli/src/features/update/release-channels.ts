import semver from "semver"

/**
 * Release channels, derived from a version's prerelease identifier: none →
 * stable, `alpha` → alpha, `beta` → beta, anything else → unknown. Publishing
 * maintains the channels with npm dist-tags (changesets pre mode publishes
 * under the pre id's tag); clients derive their channel from their own
 * running version and admit update candidates by the candidate's channel.
 */
export type ReleaseChannel = "stable" | "beta" | "alpha" | "unknown"

export const releaseChannelOf = (version: string): ReleaseChannel => {
  const identifier = semver.prerelease(version)?.[0]
  if (identifier === undefined) return "stable"
  if (identifier === "alpha" || identifier === "beta") return identifier
  return "unknown"
}

/**
 * The candidate channels a client follows: stable clients follow only stable
 * releases, beta clients also follow betas, alpha clients follow everything.
 * An unknown channel is admitted nowhere and follows stable rules as a
 * client, conservatively.
 */
export const admittedChannels = (client: ReleaseChannel): ReadonlySet<ReleaseChannel> => {
  switch (client) {
    case "alpha":
      return new Set<ReleaseChannel>(["stable", "beta", "alpha"])
    case "beta":
      return new Set<ReleaseChannel>(["stable", "beta"])
    case "stable":
    case "unknown":
      return new Set<ReleaseChannel>(["stable"])
  }
}

export const isValidVersion = (version: string): boolean =>
  semver.valid(version) !== null

export const isNewerVersion = (candidate: string, current: string): boolean =>
  isValidVersion(candidate)
  && isValidVersion(current)
  && semver.gt(candidate, current)

export const newestFirst = (
  versions: ReadonlyArray<string>,
): ReadonlyArray<string> => [...versions].sort(semver.rcompare)
