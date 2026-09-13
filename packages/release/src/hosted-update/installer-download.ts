import { Option } from "effect"
import { defaultArtifactDownloadPolicy, downloadArtifact, type ArtifactDownloadInput } from "../artifact-download"
import type { UpdateManifest } from "./manifest"

/** Keep installer responses short while retaining full-file integrity verification. */
export const downloadUpdateArtifact = (options: {
  readonly manifest: UpdateManifest
  readonly url: string
  readonly destination: string
  readonly onProgress: ArtifactDownloadInput["onProgress"]
}) => downloadArtifact({
  url: options.url, destination: options.destination,
  bytes: options.manifest.artifact.bytes, sha256: options.manifest.artifact.sha256,
  strategy: { _tag: "Segmented", concurrency: 4, chunkBytes: 4 * 1024 * 1024, fallbackToSequential: true },
  policy: { ...defaultArtifactDownloadPolicy, attemptTimeout: "1 hour", totalTimeout: "185 minutes" },
  onProgress: options.onProgress, onVerificationProgress: Option.none(),
})
