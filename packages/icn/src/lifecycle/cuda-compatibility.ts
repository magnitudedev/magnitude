import { Option } from "effect"
import type { ReleaseArtifact, ReleaseManifest } from "@magnitudedev/release"
import type { HostId } from "@magnitudedev/release/targets"

const imageApplies = (
  image: {
    readonly target: number
    readonly architectureSpecific: boolean
  },
  architecture: number,
): boolean => image.architectureSpecific
  ? architecture === image.target
  : architecture >= image.target

export const selectCudaArtifact = (
  manifest: ReleaseManifest,
  host: HostId,
  driverApi: number,
  architectures: readonly string[],
): Option.Option<ReleaseArtifact> => {
  const devices = architectures
    .map((architecture) => Number.parseInt(architecture, 10))
    .filter(Number.isFinite)
  const candidates = manifest.artifacts.flatMap((artifact) => {
    if (
      artifact.kind !== "icn-backend"
      || Option.getOrUndefined(artifact.host) !== host
      || Option.getOrUndefined(artifact.backend) !== "cuda"
      || Option.isNone(artifact.compatibility)
      || artifact.compatibility.value.kind !== "cuda"
    ) return []
    const compatibility = artifact.compatibility.value
    const usableImages = compatibility.images.filter((image) =>
      driverApi >= image.minimumDriverApi)
    const applicableImages = usableImages.filter((image) =>
      devices.some((device) => imageApplies(image, device)))
    return devices.length > 0
      && devices.every((device) => usableImages.some((image) => imageApplies(image, device)))
      ? [{
          artifact,
          architectureSpecific: applicableImages.some((image) => image.architectureSpecific),
          bestTarget: Math.max(...applicableImages.map((image) => image.target)),
          bestPtxVersion: Math.max(...applicableImages.map((image) => Number(image.ptxVersion))),
          toolkitVersion: compatibility.toolkitVersion,
        }]
      : []
  }).sort((left, right) =>
    Number(right.architectureSpecific) - Number(left.architectureSpecific)
      || right.bestTarget - left.bestTarget
      || right.bestPtxVersion - left.bestPtxVersion
      || right.toolkitVersion.localeCompare(left.toolkitVersion, undefined, { numeric: true })
      || left.artifact.id.localeCompare(right.artifact.id))
  return Option.fromNullable(candidates[0]?.artifact)
}
