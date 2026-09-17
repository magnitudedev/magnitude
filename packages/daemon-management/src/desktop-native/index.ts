import { createRequire } from "node:module"
import { join } from "node:path"
import { Context, Effect, Layer, Option, Schema, type Scope } from "effect"
import { WindowsPipeName } from "@magnitudedev/utils/windows-native"

export class NativeHostUnavailable extends Schema.TaggedError<NativeHostUnavailable>()("NativeHostUnavailable", {
  message: Schema.String,
}) {}

/** The opaque lock remains strongly reachable until its Effect scope closes. */
export interface NativeHost {
  readonly requireInteractiveDesktop: Effect.Effect<void, NativeHostUnavailable>
  readonly localAppDataDirectory: Effect.Effect<string, NativeHostUnavailable>
  readonly acquireOwnership: (path: string) => Effect.Effect<Option.Option<object>, NativeHostUnavailable, Scope.Scope>
  readonly guardParent: (descriptor: number) => Effect.Effect<void, NativeHostUnavailable>
  readonly ownedEndpoint: (lock: object, directory: string) => Effect.Effect<string, NativeHostUnavailable>
  readonly inspectEndpoint: (directory: string) => Effect.Effect<Option.Option<string>, NativeHostUnavailable>
}
export const NativeHost = Context.GenericTag<NativeHost>("@magnitudedev/daemon-management/NativeHost")

interface NativeBindings {
  readonly isInteractiveDesktop: () => unknown
  readonly localAppDataDirectory: () => unknown
  readonly acquireLock: (path: string) => object | null
  readonly releaseLock: (lock: object) => void
  readonly guardParent: (descriptor: number) => void
  readonly lockEndpoint: (lock: object) => unknown
  readonly inspectApplicationEndpoint: (directory: string) => unknown
}

/** Privileged composition roots supply the installed, signed addon path. */
export const nativeHostLayerFromLoader = (load: () => unknown): Layer.Layer<NativeHost, NativeHostUnavailable> => Layer.effect(
  NativeHost,
  Effect.gen(function* () {
    const bindings = yield* Effect.try({
      try: () => load() as NativeBindings,
      catch: error => new NativeHostUnavailable({ message: `Cannot load native host: ${String(error)}` }),
    })
    return NativeHost.of({
      requireInteractiveDesktop: Effect.try({
        try: () => bindings.isInteractiveDesktop(),
        catch: () => new NativeHostUnavailable({ message: "Cannot inspect the Windows desktop session." }),
      }).pipe(
        Effect.flatMap(Schema.decodeUnknown(Schema.Boolean)),
        Effect.mapError(() => new NativeHostUnavailable({ message: "Cannot inspect the Windows desktop session." })),
        Effect.flatMap(available => available ? Effect.void : Effect.fail(new NativeHostUnavailable({
          message: "Magnitude requires a graphical Windows session. Start the desktop app in your desktop session; headless commands can then control it.",
        }))),
      ),
      localAppDataDirectory: Effect.try({
        try: () => bindings.localAppDataDirectory(),
        catch: error => new NativeHostUnavailable({ message: String(error) }),
      }).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.NonEmptyString)), Effect.mapError(error => new NativeHostUnavailable({ message: String(error) }))),
      acquireOwnership: path => Effect.acquireRelease(
        Effect.try({
          try: () => Option.fromNullable(bindings.acquireLock(path)),
          catch: error => new NativeHostUnavailable({ message: String(error) }),
        }),
        lock => Effect.sync(() => { if (Option.isSome(lock)) bindings.releaseLock(lock.value) }),
      ),
      guardParent: descriptor => Effect.try({
        try: () => bindings.guardParent(descriptor),
        catch: error => new NativeHostUnavailable({ message: String(error) }),
      }),
      ownedEndpoint: (lock, directory) => process.platform !== "win32" ? Effect.succeed(join(directory, "application.sock")) : Effect.try({
        try: () => bindings.lockEndpoint(lock), catch: error => new NativeHostUnavailable({ message: String(error) }),
      }).pipe(Effect.flatMap(Schema.decodeUnknown(WindowsPipeName)), Effect.mapError(error => new NativeHostUnavailable({ message: String(error) }))),
      inspectEndpoint: directory => process.platform !== "win32" ? Effect.succeed(Option.some(join(directory, "application.sock"))) : Effect.try({
        try: () => bindings.inspectApplicationEndpoint(directory), catch: error => new NativeHostUnavailable({ message: String(error) }),
      }).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.NullOr(WindowsPipeName))), Effect.map(Option.fromNullable), Effect.mapError(error => new NativeHostUnavailable({ message: String(error) }))),
    })
  }),
)

export const nativeHostLayer = (addonPath: string) => nativeHostLayerFromLoader(() => createRequire(import.meta.url)(addonPath))
