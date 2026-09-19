import { HttpClient } from "@effect/platform"
import { artifactDeliveryClient, type ArtifactDelivery } from "./artifact-delivery"
import { Effect, Option } from "effect"
import { defaultArtifactDownloadPolicy, downloadArtifact, type ArtifactDownloadInput } from "../artifact-download"
import type { UpdateRelease } from "./release"

/** Keep installer responses short while retaining full-file integrity verification. */
export const downloadUpdateArtifact = (options: {
  readonly release: UpdateRelease
  readonly url: string
  readonly destination: string
  readonly onProgress: ArtifactDownloadInput["onProgress"]
  readonly artifactDelivery?: ArtifactDelivery
}) => Effect.gen(function* () {
  const client = yield* artifactDeliveryClient(options.artifactDelivery)
  return yield* downloadArtifact({
  url: options.url, destination: options.destination,
  bytes: options.release.bytes, sha256: options.release.sha256,
  strategy: { _tag: "Segmented", concurrency: 4, chunkBytes: 4 * 1024 * 1024, fallbackToSequential: true },
  policy: { ...defaultArtifactDownloadPolicy, attemptTimeout: "1 hour", totalTimeout: "185 minutes" },
  onProgress: options.onProgress, onVerificationProgress: Option.none(),
}).pipe(Effect.provideService(HttpClient.HttpClient, client))
}).pipe(Effect.tapError(error => Effect.logDebug("Installer transfer failed", { phase: error.phase, message: error.message })))
