import { Context, Effect, Layer, Option, Schema, Scope } from "effect"
import { createRequire } from "node:module"

export const MacFileIdentity = Schema.String.pipe(Schema.pattern(/^\d{1,20}:\d{1,20}$/), Schema.brand("MacFileIdentity"))
export type MacFileIdentity = typeof MacFileIdentity.Type
export class MacUpdateFilesystemFailed extends Schema.TaggedError<MacUpdateFilesystemFailed>()("MacUpdateFilesystemFailed", {}) {}
const handle = Symbol("MacUpdateDirectory")
export interface MacUpdateDirectory {
  readonly identity: MacFileIdentity
  readonly path: string
  readonly [handle]: object
}
export interface MacUpdateFilesystem {
  readonly open: (path: string, privateDirectory: boolean) => Effect.Effect<MacUpdateDirectory, MacUpdateFilesystemFailed, Scope.Scope>
  readonly inspect: (directory: MacUpdateDirectory, name: string) => Effect.Effect<Option.Option<MacFileIdentity>, MacUpdateFilesystemFailed>
  readonly removeRecord: (directory: MacUpdateDirectory, expected: Uint8Array) => Effect.Effect<void, MacUpdateFilesystemFailed>
  readonly readRecord: (directory: MacUpdateDirectory) => Effect.Effect<Option.Option<Uint8Array>, MacUpdateFilesystemFailed>
  readonly removeEmptyDirectory: (parent: MacUpdateDirectory, name: string, child: MacUpdateDirectory) => Effect.Effect<boolean, MacUpdateFilesystemFailed>
  readonly sync: (directory: MacUpdateDirectory) => Effect.Effect<void, MacUpdateFilesystemFailed>
  readonly removeTree: (directory: MacUpdateDirectory, name: string, expected: MacFileIdentity) => Effect.Effect<void, MacUpdateFilesystemFailed>
  readonly syncTree: (directory: MacUpdateDirectory, name: string, expected: MacFileIdentity) => Effect.Effect<void, MacUpdateFilesystemFailed>
  readonly writeRecord: (directory: MacUpdateDirectory, bytes: Uint8Array) => Effect.Effect<void, MacUpdateFilesystemFailed>
  readonly exchange: (installed: MacUpdateDirectory, installedName: string, previous: MacFileIdentity,
    staging: MacUpdateDirectory, stagedName: string, replacement: MacFileIdentity) => Effect.Effect<void, MacUpdateFilesystemFailed>
  readonly publish: (installed: MacUpdateDirectory, installedName: string,
    staging: MacUpdateDirectory, stagedName: string, replacement: MacFileIdentity) => Effect.Effect<void, MacUpdateFilesystemFailed>
}
export const MacUpdateFilesystem = Context.GenericTag<MacUpdateFilesystem>("@magnitudedev/daemon-management/MacUpdateFilesystem")

interface Bindings {
  readonly openMacUpdateDirectory: (path: string, privateDirectory: boolean) => { readonly identity: unknown; readonly path: unknown }
  readonly closeMacUpdateDirectory: (directory: object) => void
  readonly removeEmptyMacUpdateDirectory: (parent: object, name: string, child: object) => boolean
  readonly syncMacUpdateDirectory: (directory: object) => void
  readonly removeMacUpdateTree: (directory: object, name: string, expected: string) => void
  readonly syncMacUpdateTree: (directory: object, name: string, expected: string) => void
  readonly inspectMacUpdateDirectory: (directory: object, name: string) => unknown
  readonly removeMacUpdateRecord: (directory: object, expected: Buffer) => void
  readonly readMacUpdateRecord: (directory: object) => Uint8Array | null
  readonly writeMacUpdateRecord: (directory: object, bytes: Buffer) => void
  readonly exchangeMacUpdateDirectories: (installed: object, installedName: string, previous: string,
    staging: object, stagedName: string, replacement: string) => void
  readonly publishMacUpdateDirectory: (installed: object, installedName: string,
    staging: object, stagedName: string, replacement: string) => void
}
const attempt = <A>(run: () => A) => Effect.try({ try: run, catch: () => new MacUpdateFilesystemFailed() })

/** Installer-process primitives; callers retain exclusion and reconcile identities after any exchange error. */
export const nativeMacUpdateFilesystem = (addonPath: string) => Layer.effect(MacUpdateFilesystem, Effect.gen(function* () {
  const native = yield* attempt(() => createRequire(import.meta.url)(addonPath) as Bindings)
  return MacUpdateFilesystem.of({
    open: (path, privateDirectory) => Effect.gen(function* () {
      const retained = yield* Effect.acquireRelease(attempt(() => native.openMacUpdateDirectory(path, privateDirectory)),
        directory => Effect.sync(() => native.closeMacUpdateDirectory(directory)))
      const identity = yield* Schema.decodeUnknown(MacFileIdentity)(retained.identity).pipe(Effect.mapError(() => new MacUpdateFilesystemFailed()))
      const canonicalPath = yield* Schema.decodeUnknown(Schema.String.pipe(Schema.startsWith("/")))(retained.path).pipe(Effect.mapError(() => new MacUpdateFilesystemFailed()))
      return { identity, path: canonicalPath, [handle]: retained }
    }),
    inspect: (directory, name) => attempt(() => native.inspectMacUpdateDirectory(directory[handle], name)).pipe(
      Effect.flatMap(Schema.decodeUnknown(Schema.OptionFromNullOr(MacFileIdentity))),
      Effect.mapError(() => new MacUpdateFilesystemFailed())),
    removeRecord: (directory, expected) => attempt(() => native.removeMacUpdateRecord(directory[handle], Buffer.from(expected))),
    readRecord: directory => attempt(() => Option.fromNullable(native.readMacUpdateRecord(directory[handle]))),
    removeEmptyDirectory: (parent, name, child) => attempt(() => native.removeEmptyMacUpdateDirectory(parent[handle], name, child[handle])),
    sync: directory => attempt(() => native.syncMacUpdateDirectory(directory[handle])),
    removeTree: (directory, name, expected) => attempt(() => native.removeMacUpdateTree(directory[handle], name, expected)),
    syncTree: (directory, name, expected) => attempt(() => native.syncMacUpdateTree(directory[handle], name, expected)),
    writeRecord: (directory, bytes) => attempt(() => native.writeMacUpdateRecord(directory[handle], Buffer.from(bytes))),
    exchange: (installed, installedName, previous, staging, stagedName, replacement) =>
      attempt(() => native.exchangeMacUpdateDirectories(installed[handle], installedName, previous, staging[handle], stagedName, replacement)),
    publish: (installed, installedName, staging, stagedName, replacement) =>
      attempt(() => native.publishMacUpdateDirectory(installed[handle], installedName, staging[handle], stagedName, replacement)),
  })
}))
