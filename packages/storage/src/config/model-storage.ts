import * as FileSystem from '@effect/platform/FileSystem'
import { Effect, Option } from 'effect'
import { isAbsolute, join, normalize } from 'node:path'

/** Where downloaded models live when `modelsDirectory` is not configured. */
export const defaultModelStoreRoot = (dataDir: string): string => join(dataDir, 'models')

export type ModelStoreSource = 'Default' | 'Configured'

export interface ModelStoreSelection {
  /** The configured or default path, exactly as the user would see it. */
  readonly path: string
  readonly source: ModelStoreSource
  /** Why a configured value was not used. */
  readonly warning: Option.Option<string>
}

export interface ModelStoreLocation extends ModelStoreSelection {
  /** The absolute path handed to the inference engine, with a symlinked root resolved. */
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
