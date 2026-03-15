import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

import type { CachedCatalogSourceData } from './contracts'
import type { GlobalStorageShape } from '../services'

export const MODELS_DEV_TTL_MS = 60 * 60 * 1000
export const OPENROUTER_TTL_MS = 15 * 60 * 1000
export const STALE_GRACE_MS = 7 * 24 * 60 * 60 * 1000

function getCatalogCacheDir(globalStorage: GlobalStorageShape): string {
  return join(globalStorage.root, 'model-catalog')
}

function getCatalogCachePath(globalStorage: GlobalStorageShape, sourceId: string): string {
  return join(getCatalogCacheDir(globalStorage), `${sourceId}.json`)
}

export async function loadCatalogCache<T>(
  globalStorage: GlobalStorageShape,
  sourceId: string
): Promise<CachedCatalogSourceData<T> | null> {
  try {
    const raw = await readFile(getCatalogCachePath(globalStorage, sourceId), 'utf8')
    return JSON.parse(raw) as CachedCatalogSourceData<T>
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return null
    throw error
  }
}

export async function saveCatalogCache(
  globalStorage: GlobalStorageShape,
  sourceId: string,
  data: unknown,
  ttlMs: number
): Promise<void> {
  await mkdir(getCatalogCacheDir(globalStorage), { recursive: true })
  const payload: CachedCatalogSourceData<unknown> = {
    _cachedAt: Date.now(),
    ttlMs,
    data,
  }
  await writeFile(getCatalogCachePath(globalStorage, sourceId), JSON.stringify(payload), 'utf8')
}

export function isCatalogCacheValid(cached: CachedCatalogSourceData<unknown>): boolean {
  return Date.now() - cached._cachedAt < cached.ttlMs
}

export function isCatalogCacheStale(cached: CachedCatalogSourceData<unknown>): boolean {
  return Date.now() - cached._cachedAt < STALE_GRACE_MS
}