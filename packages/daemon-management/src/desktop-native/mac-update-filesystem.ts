import { Context, Effect, Layer, Option, Schema, Scope } from "effect"
import { createRequire } from "node:module"

export const MacFileIdentity = Schema.String.pipe(Schema.pattern(/^\d{1,20}:\d{1,20}$/), Schema.brand("MacFileIdentity"))
export type MacFileIdentity = typeof MacFileIdentity.Type
export class MacUpdateFilesystemFailed extends Schema.TaggedError<MacUpdateFilesystemFailed>()("MacUpdateFilesystemFailed", {}) {}
const handle = Symbol("MacUpdateDirectory")
export interface MacUpdateDirectory {
  readonly identity: MacFileIdentity
  readonly [handle]: object
}
export interface MacUpdateFilesystem {
  readonly open: (path: string, privateDirectory: boolean) => Effect.Effect<MacUpdateDirectory, MacUpdateFilesystemFailed, Scope.Scope>
  readonly inspect: (directory: MacUpdateDirectory, name: string) => Effect.Effect<Option.Option<MacFileIdentity>, MacUpdateFilesystemFailed>
  readonly readRecord: (directory: MacUpdateDirectory) => Effect.Effect<Option.Option<Uint8Array>, MacUpdateFilesystemFailed>
  readonly writeRecord: (directory: MacUpdateDirectory, bytes: Uint8Array) => Effect.Effect<void, MacUpdateFilesystemFailed>
  readonly exchange: (installed: MacUpdateDirectory, installedName: string, previous: MacFileIdentity,
    staging: MacUpdateDirectory, stagedName: string, replacement: MacFileIdentity) => Effect.Effect<void, MacUpdateFilesystemFailed>
}
export const MacUpdateFilesystem = Context.GenericTag<MacUpdateFilesystem>("@magnitudedev/daemon-management/MacUpdateFilesystem")

interface Bindings {
  readonly openMacUpdateDirectory: (path: string, privateDirectory: boolean) => { readonly identity: unknown }
  readonly closeMacUpdateDirectory: (directory: object) => void
  readonly inspectMacUpdateDirectory: (directory: object, name: string) => unknown
  readonly readMacUpdateRecord: (directory: object) => Uint8Array | null
  readonly writeMacUpdateRecord: (directory: object, bytes: Buffer) => void
  readonly exchangeMacUpdateDirectories: (installed: object, installedName: string, previous: string,
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
      return { identity, [handle]: retained }
    }),
    inspect: (directory, name) => attempt(() => native.inspectMacUpdateDirectory(directory[handle], name)).pipe(
      Effect.flatMap(Schema.decodeUnknown(Schema.OptionFromNullOr(MacFileIdentity))),
      Effect.mapError(() => new MacUpdateFilesystemFailed())),
    readRecord: directory => attempt(() => Option.fromNullable(native.readMacUpdateRecord(directory[handle]))),
    writeRecord: (directory, bytes) => attempt(() => native.writeMacUpdateRecord(directory[handle], Buffer.from(bytes))),
    exchange: (installed, installedName, previous, staging, stagedName, replacement) =>
      attempt(() => native.exchangeMacUpdateDirectories(installed[handle], installedName, previous, staging[handle], stagedName, replacement)),
  })
}))
