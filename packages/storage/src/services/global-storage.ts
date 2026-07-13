import { Context, Layer } from 'effect'

import {
  defaultGlobalStorageRoot,
  makeGlobalStoragePaths,
  type GlobalStoragePaths,
} from '../paths'

export interface GlobalStorageShape {
  readonly root: string
  readonly paths: GlobalStoragePaths
}

export class GlobalStorage extends Context.Tag('GlobalStorage')<
  GlobalStorage,
  GlobalStorageShape
>() {}

export function makeGlobalStorage(options?: {
  readonly root?: string
}): GlobalStorageShape {
  const root = options?.root ?? defaultGlobalStorageRoot()

  return {
    root,
    paths: makeGlobalStoragePaths(root),
  }
}

export const GlobalStorageLive = Layer.succeed(
  GlobalStorage,
  GlobalStorage.of(makeGlobalStorage())
)