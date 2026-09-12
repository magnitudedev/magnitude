import { FetchHttpClient, FileSystem } from "@effect/platform"
import { NodeContext } from "@effect/platform-node"
import { defaultArtifactDownloadPolicy, downloadArtifact } from "@magnitudedev/release"
import { checkHostedUpdate, resolveHostedDownload, type HostedUpdateConnection } from "@magnitudedev/release/hosted-update"
import { Effect, Option } from "effect"
import type { KeyObject } from "node:crypto"
import { basename, join } from "node:path"
import { ApplicationUpdateFailed, ApplicationUpdateSource } from "./application-update"
import { NativeMacUpdate, stageMacUpdateArchive } from "./mac-update-stage"
import { ApplicationUpdateHandoff } from "./update-handoff"

export const macUpdateSource = (options: HostedUpdateConnection & {
  readonly trustedPublishers: ReadonlyMap<string, KeyObject>
  readonly storageOrigin: string
  readonly cacheDirectory: string
}) => Effect.gen(function* () {
  const native = yield* NativeMacUpdate
  const handoff = yield* ApplicationUpdateHandoff
  return ApplicationUpdateSource.of({
    check: checkHostedUpdate(options).pipe(
      Effect.mapError(() => new ApplicationUpdateFailed({ message: "Could not check for application updates." })), Effect.provide(FetchHttpClient.layer),
    ),
    download: (candidate, progress) => Effect.gen(function* () {
      const url = yield* resolveHostedDownload({ ...options, manifest: candidate })
      const fs = yield* FileSystem.FileSystem
      yield* fs.makeDirectory(options.cacheDirectory, { recursive: true, mode: 0o700 })
      const directory = yield* fs.makeTempDirectoryScoped({ directory: options.cacheDirectory, prefix: "desktop-update-" })
      const downloaded = yield* downloadArtifact({
        url, destination: join(directory, basename(candidate.artifact.path)),
        bytes: candidate.artifact.bytes, sha256: candidate.artifact.sha256,
        strategy: { _tag: "Sequential" }, policy: defaultArtifactDownloadPolicy,
        onProgress: Option.some(value => progress(value.acceptedBytes)), onVerificationProgress: Option.none(),
      })
      return downloaded.destination
    }).pipe(Effect.mapError(() => new ApplicationUpdateFailed({ message: "Could not download and verify the application update." })), Effect.provide([NodeContext.layer, FetchHttpClient.layer])),
    stage: (archive, candidate) => handoff.record(candidate.version).pipe(
      Effect.zipRight(stageMacUpdateArchive(archive).pipe(Effect.provideService(NativeMacUpdate, native))),
      Effect.mapError(error => new ApplicationUpdateFailed({ message: error.message })),
    ),
  })
})
