import * as Path from "@effect/platform/Path"
import { Effect, Option, Schema } from "effect"
import { acquireRelease } from "../acquisition"
import { ReleaseArtifactSchema } from "../contracts"
import { desktopUpdateArchive } from "../targets"
import { findReleaseUpdate, UpdateDiscoveryFailed } from "./discovery"

export const DesktopUpdateCandidate = Schema.Struct({
  version: Schema.NonEmptyString,
  artifact: ReleaseArtifactSchema,
})

/** A desktop update requires its own matched archive; a CLI-only release is insufficient. */
export const findMacDesktopUpdate = (options: {
  readonly currentVersion: string
  readonly host: "darwin-arm64" | "darwin-x64"
  readonly registryUrl: string
  readonly releaseBaseUrl: string
  readonly cacheDirectory: string
}) => Effect.gen(function* () {
  const path = yield* Path.Path
  return yield* findReleaseUpdate({
    currentVersion: options.currentVersion,
    registryUrl: options.registryUrl,
    verify: version => Effect.gen(function* () {
      const { manifest } = yield* acquireRelease(options.releaseBaseUrl, version,
        path.join(options.cacheDirectory, version))
      const archives = manifest.artifacts.filter(artifact =>
        artifact.kind === "desktop" && Option.contains(artifact.host, options.host) &&
        artifact.id === `desktop-update-${options.host}` && artifact.filename === desktopUpdateArchive(options.host))
      if (archives.length !== 1) return yield* new UpdateDiscoveryFailed({
        stage: "release", reason: "Release does not contain the matching Mac desktop update archive",
      })
      return DesktopUpdateCandidate.make({ version: manifest.version, artifact: archives[0]! })
    }),
  })
})
