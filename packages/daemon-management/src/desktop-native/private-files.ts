import { FileSystem } from "@effect/platform"
import { Context, Effect, Layer, Schema } from "effect"
import { createRequire } from "node:module"

export class PrivateFilePermissionsFailed extends Schema.TaggedError<PrivateFilePermissionsFailed>()("PrivateFilePermissionsFailed", {}) {}
export interface PrivateFilePermissions {
  readonly prepareDirectory: (path: string) => Effect.Effect<void, PrivateFilePermissionsFailed>
  readonly createFile: (path: string) => Effect.Effect<void, PrivateFilePermissionsFailed>
  readonly protectFile: (path: string) => Effect.Effect<void, PrivateFilePermissionsFailed>
}
export const PrivateFilePermissions = Context.GenericTag<PrivateFilePermissions>("@magnitudedev/daemon-management/PrivateFilePermissions")

export const unixPrivateFilePermissions = Layer.effect(PrivateFilePermissions, Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  return PrivateFilePermissions.of({
    prepareDirectory: path => fs.makeDirectory(path, { recursive: true, mode: 0o700 }).pipe(
      Effect.zipRight(fs.chmod(path, 0o700)), Effect.mapError(() => new PrivateFilePermissionsFailed())),
    protectFile: path => fs.chmod(path, 0o600).pipe(Effect.mapError(() => new PrivateFilePermissionsFailed())),
    createFile: path => fs.writeFileString(path, "", { flag: "wx", mode: 0o600 }).pipe(Effect.mapError(() => new PrivateFilePermissionsFailed())),
  })
}))

interface WindowsBindings {
  readonly preparePrivateDirectory: (path: string) => void
  readonly createPrivateContent: (path: string) => void
  readonly validatePrivateContent: (path: string) => void
}

/** Explicit ownership also works when an elevated token's default owner is Administrators. */
export const windowsPrivateFilePermissions = (addonPath: string) => Layer.effect(PrivateFilePermissions, Effect.gen(function* () {
  const bindings = yield* Effect.try({
    try: () => createRequire(import.meta.url)(addonPath) as WindowsBindings,
    catch: () => new PrivateFilePermissionsFailed(),
  })
  return PrivateFilePermissions.of({
    prepareDirectory: path => Effect.try({ try: () => bindings.preparePrivateDirectory(path), catch: () => new PrivateFilePermissionsFailed() }),
    createFile: path => Effect.try({ try: () => bindings.createPrivateContent(path), catch: () => new PrivateFilePermissionsFailed() }),
    protectFile: path => Effect.try({ try: () => bindings.validatePrivateContent(path), catch: () => new PrivateFilePermissionsFailed() }),
  })
}))
