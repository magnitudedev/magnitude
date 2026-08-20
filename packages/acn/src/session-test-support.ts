/** Test-only helpers: real Magnitude storage and filesystem layers over a temp root. */
import { join } from "node:path"
import { BunCommandExecutor, BunFileSystem, BunPath } from "@effect/platform-bun"
import { Layer } from "effect"
import {
  GlobalStorage,
  makeGlobalStorage,
  ProjectStorageLiveFromCwd,
  StorageLive,
  VersionLive,
  type MagnitudeStorage,
} from "@magnitudedev/storage"
import { FileSystemManager, FileSystemManagerLive } from "./file-system-manager"
import { SessionInspector, SessionInspectorLive } from "./session-inspector"

export const testPlatformLayer = Layer.mergeAll(
  BunFileSystem.layer,
  BunPath.layer,
  BunCommandExecutor.layer.pipe(Layer.provide(BunFileSystem.layer)),
)

/** Real storage rooted in a private temp directory. */
export const makeTestStorageLayer = (root: string): Layer.Layer<MagnitudeStorage> =>
  StorageLive.pipe(
    Layer.orDie,
    Layer.provide(Layer.mergeAll(
      VersionLive("0.0.0-test"),
      ProjectStorageLiveFromCwd(root),
      Layer.succeed(GlobalStorage, GlobalStorage.of(makeGlobalStorage({
        root: join(root, ".magnitude-test-data"),
      }))),
      testPlatformLayer,
    )),
  )

export const testFileSystemManagerLayer: Layer.Layer<FileSystemManager> =
  FileSystemManagerLive.pipe(Layer.provide(testPlatformLayer))

export const makeTestSessionInspectorLayer = (
  storage: Layer.Layer<MagnitudeStorage>,
): Layer.Layer<SessionInspector> => SessionInspectorLive.pipe(Layer.provide(storage))
