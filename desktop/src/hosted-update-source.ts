import { FetchHttpClient, FileSystem } from "@effect/platform"
import { NodeContext } from "@effect/platform-node"
import { checkHostedUpdate, resolveHostedDownload, downloadUpdateArtifact, updateInstallerFilename, ReleaseTarget, type HostedUpdateConnection } from "@magnitudedev/release/hosted-update"
import { Effect, Option, Schema } from "effect"
import type { KeyObject } from "node:crypto"
import { join } from "node:path"
import { ApplicationUpdateFailed, ApplicationUpdateSource } from "./application-update"

export type HostedUpdateSourceOptions = HostedUpdateConnection & {
  readonly trustedPublishers: ReadonlyMap<string, KeyObject>
  readonly cacheDirectory: string
}

/** All native installers consume the same authenticated, checksum-verified transfer. */
export const hostedUpdateSource = (options: HostedUpdateSourceOptions, stage: ApplicationUpdateSource["stage"]): ApplicationUpdateSource => ({
  check: checkHostedUpdate(options).pipe(
    Effect.mapError(() => new ApplicationUpdateFailed({ message: "Could not check for application updates." })), Effect.provide(FetchHttpClient.layer),
  ),
  download: (candidate, progress) => Effect.gen(function* () {
    const url = yield* resolveHostedDownload({ ...options, release: candidate })
    const fs = yield* FileSystem.FileSystem
    yield* fs.makeDirectory(options.cacheDirectory, { recursive: true, mode: 0o700 })
    const directory = yield* fs.makeTempDirectoryScoped({ directory: options.cacheDirectory, prefix: "desktop-update-" })
    const downloaded = yield* downloadUpdateArtifact({
      url, destination: join(directory, updateInstallerFilename(yield* Schema.decodeUnknown(ReleaseTarget)({ os: options.metadata.os, arch: options.metadata.arch, package: options.metadata.package }))),
      release: candidate,
      ...(options.artifactDelivery ? { artifactDelivery: options.artifactDelivery } : {}),
      onProgress: Option.some(value => progress(value.acceptedBytes)),
    })
    return downloaded.destination
  }).pipe(Effect.mapError(() => new ApplicationUpdateFailed({ message: "Could not download and verify the application update." })), Effect.provide([NodeContext.layer, FetchHttpClient.layer])),
  stage,
})
