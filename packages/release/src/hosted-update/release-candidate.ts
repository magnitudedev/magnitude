import { Effect, Option, Schema } from "effect"
import type { ReleaseArtifact, ReleaseManifest } from "../contracts"
import { releaseHosts, type HostId } from "../targets"
import { UpdateManifest } from "./manifest"

export class HostedCandidateInvalid extends Schema.TaggedError<HostedCandidateInvalid>()("HostedCandidateInvalid", {
  message: Schema.String,
}) {}

/** Hosted desktop publication is a projection of the complete accepted native release. */
export const hostedDesktopManifests = (
  release: Pick<ReleaseManifest, "version" | "sourceCommit"> & { readonly artifacts: readonly ReleaseArtifact[] },
  expectedCommit: string,
) => Effect.gen(function* () {
  if (release.sourceCommit !== expectedCommit) return yield* new HostedCandidateInvalid({ message: "Public release differs from the selected source" })
  const expected = new Map<string, { host: HostId; target: typeof UpdateManifest.Type.artifact.target }>()
  for (const host of releaseHosts) {
    const arch = host.id.includes("arm64") ? "arm64" : "x64"
    if (host.id.startsWith("darwin-")) {
      expected.set(`desktop-${host.id}`, { host: host.id, target: { os: "darwin", arch, package: "dmg" } })
      expected.set(`desktop-update-${host.id}`, { host: host.id, target: { os: "darwin", arch, package: "mac-zip" } })
    } else if (host.id.startsWith("linux-")) {
      for (const format of ["deb", "rpm"] as const) expected.set(`desktop-${host.id}-${format}`, { host: host.id, target: { os: "linux", arch, package: format } })
    } else return yield* new HostedCandidateInvalid({ message: "Configured release host has no accepted desktop publication contract" })
  }
  const desktops = release.artifacts.filter(artifact => artifact.kind === "desktop")
  if (desktops.length !== expected.size) return yield* new HostedCandidateInvalid({ message: "Public release is missing the complete desktop artifact graph" })
  const manifests = []
  for (const artifact of desktops) {
    const entry = expected.get(artifact.id)
    if (!entry || Option.getOrUndefined(artifact.host) !== entry.host) return yield* new HostedCandidateInvalid({ message: "Unexpected desktop artifact identity or host" })
    expected.delete(artifact.id)
    const suffix = entry.target.package === "mac-zip" ? ".zip" : `.${entry.target.package}`
    if (!artifact.filename.endsWith(suffix)) return yield* new HostedCandidateInvalid({ message: "Desktop artifact filename differs from its package format" })
    manifests.push(yield* Schema.decodeUnknown(UpdateManifest)({
      protocol: 1, version: release.version, commit: release.sourceCommit,
      artifact: { id: artifact.id, target: entry.target, path: `releases/${release.version}/${artifact.filename}`, bytes: artifact.bytes, sha256: artifact.sha256 },
    }).pipe(Effect.mapError(() => new HostedCandidateInvalid({ message: "Desktop artifact cannot be represented by the hosted release contract" }))))
  }
  return manifests
})
