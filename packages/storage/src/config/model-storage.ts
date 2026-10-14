import * as FileSystem from '@effect/platform/FileSystem'
import { Effect, Option } from 'effect'
import { isAbsolute, join, normalize } from 'node:path'

/** Where downloaded models live when `modelsDirectory` is not configured. */
export const defaultModelStoreRoot = (dataDir: string): string => join(dataDir, 'models')

export type ModelStoreSource = 'Default' | 'Configured'

export interface ModelStoreSelection {
  readonly path: string
  readonly source: ModelStoreSource
  readonly warning: Option.Option<string>
}

export interface ModelStoreLocation extends ModelStoreSelection {
  readonly root: string
}

/**
 * Pure selection of the store path. A configured value must be absolute; anything else falls
 * back to the default with a warning so a bad edit never prevents the service from starting.
 */
export const selectModelStorePath = (
  dataDir: string,
  configured: Option.Option<string>,
): ModelStoreSelection => {
  const fallback = defaultModelStoreRoot(dataDir)
  if (Option.isNone(configured)) return { path: fallback, source: 'Default', warning: Option.none() }
  const value = configured.value.trim()
  if (value.length === 0 || !isAbsolute(value)) {
    return {
      path: fallback,
      source: 'Default',
      warning: Option.some(`Ignoring modelsDirectory "${configured.value}": it must be an absolute path. Using ${fallback}.`),
    }
  }
  return { path: normalize(value), source: 'Configured', warning: Option.none() }
}

/**
 * Resolves the selected path to what the engine should open. The engine refuses a store root
 * that is itself a symbolic link, so the link is followed here once, at startup. A path that
 * does not exist yet is kept as-is; the engine creates it. A path that exists but is not a
 * directory falls back to the default.
 */
export const resolveModelStoreLocation = (
  dataDir: string,
  configured: Option.Option<string>,
): Effect.Effect<ModelStoreLocation, never, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const realDirectory = (path: string) =>
      fs.realPath(path).pipe(
        Effect.flatMap(real => fs.stat(real).pipe(Effect.map(info => (info.type === 'Directory' ? Option.some(real) : Option.none())))),
        Effect.catchTag('SystemError', error => (error.reason === 'NotFound' ? Effect.succeed(Option.some(path)) : Effect.fail(error))),
        Effect.catchAll(() => Effect.succeed(Option.some(path))),
      )
    const selection = selectModelStorePath(dataDir, configured)
    const resolved = yield* realDirectory(selection.path)
    if (Option.isSome(resolved)) return { ...selection, root: resolved.value }
    const fallback = defaultModelStoreRoot(dataDir)
    const fallbackRoot = yield* realDirectory(fallback)
    return {
      path: fallback,
      source: 'Default',
      warning: Option.some(`Ignoring modelsDirectory "${selection.path}": it exists but is not a directory. Using ${fallback}.`),
      root: Option.getOrElse(fallbackRoot, () => fallback),
    }
  })

/** Whether this process can create the folder and write files in it, which the engine needs at startup. */
export const canWriteModelStore = (path: string): Effect.Effect<boolean, never, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    yield* fs.makeDirectory(path, { recursive: true })
    const probe = join(path, `.magnitude-write-check-${process.pid}`)
    yield* fs.writeFileString(probe, '')
    yield* fs.remove(probe)
  }).pipe(Effect.as(true), Effect.orElseSucceed(() => false))

/**
 * The location the engine is started with. A configured folder this process cannot write to
 * falls back to the default with a warning, so a bad choice never stops the service from starting.
 */
export const resolveUsableModelStoreLocation = (
  dataDir: string,
  configured: Option.Option<string>,
): Effect.Effect<ModelStoreLocation, never, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const location = yield* resolveModelStoreLocation(dataDir, configured)
    if (location.source === 'Default' || (yield* canWriteModelStore(location.root))) return location
    const fallback = yield* resolveModelStoreLocation(dataDir, Option.none())
    return {
      ...fallback,
      warning: Option.some(`Ignoring modelsDirectory "${location.path}": Magnitude can't write to it. Using ${fallback.path}.`),
    }
  })
