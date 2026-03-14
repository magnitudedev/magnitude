/**
 * Dynamic model fetching from models.dev API.
 *
 * Fetches the full model registry, caches locally, and populates
 * the provider registry with up-to-date model lists.
 *
 * Cache: <global-storage>/models-cache.json (60-minute TTL)
 * Fallback chain: valid cache → network → stale cache → built-in fallbacks
 */

import * as fs from 'node:fs'
import { defaultGlobalStorageRoot } from '@magnitudedev/storage'
import { logger } from '@magnitudedev/logger'
import { populateModels } from './registry'
import type { ModelDefinition } from './types'

// ---------------------------------------------------------------------------
// models.dev API types
// ---------------------------------------------------------------------------

interface ModelsDevModel {
  id: string
  name: string
  family?: string
  tool_call: boolean
  reasoning: boolean
  attachment: boolean
  temperature: boolean
  release_date?: string
  status?: string
  cost?: { input: number; output: number; cache_read?: number; cache_write?: number }
  limit?: { context: number; output: number; input?: number }
}

interface ModelsDevProvider {
  id: string
  name: string
  env: string[]
  npm?: string
  api?: string
  models: Record<string, ModelsDevModel>
}

type ModelsDevResponse = Record<string, ModelsDevProvider>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const API_URL = 'https://models.dev/api.json'
const CACHE_DIR = defaultGlobalStorageRoot()
const CACHE_PATH = `${CACHE_DIR}/models-cache.json`
const CACHE_TTL_MS = 60 * 60 * 1000 // 60 minutes
const FETCH_TIMEOUT_MS = 10_000       // 10 seconds
const REFRESH_INTERVAL_MS = 60 * 60 * 1000 // 1 hour

// ---------------------------------------------------------------------------
// Cache I/O
// ---------------------------------------------------------------------------

interface CachedData {
  _cachedAt: number
  data: ModelsDevResponse
}

function loadCache(): CachedData | null {
  try {
    const raw = fs.readFileSync(CACHE_PATH, 'utf-8')
    const parsed = JSON.parse(raw)
    if (parsed && typeof parsed._cachedAt === 'number' && parsed.data) {
      return parsed as CachedData
    }
  } catch {
    // Cache missing or corrupt — that's fine
  }
  return null
}

function saveCache(data: ModelsDevResponse): void {
  try {
    if (!fs.existsSync(CACHE_DIR)) {
      fs.mkdirSync(CACHE_DIR, { recursive: true })
    }
    const cached: CachedData = { _cachedAt: Date.now(), data }
    fs.writeFileSync(CACHE_PATH, JSON.stringify(cached), 'utf-8')
  } catch (err) {
    logger.warn({ err }, 'Failed to write models cache')
  }
}

function isCacheValid(cached: CachedData): boolean {
  return (Date.now() - cached._cachedAt) < CACHE_TTL_MS
}

// ---------------------------------------------------------------------------
// Network fetch
// ---------------------------------------------------------------------------

async function fetchFromNetwork(): Promise<ModelsDevResponse | null> {
  try {
    const response = await fetch(API_URL, {
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
      headers: { 'User-Agent': 'magnitude-agent' },
    })
    if (!response.ok) {
      logger.warn({ status: response.status }, 'models.dev API returned non-OK status')
      return null
    }
    const data = await response.json() as ModelsDevResponse
    return data
  } catch (err) {
    logger.warn({ err }, 'Failed to fetch from models.dev')
    return null
  }
}

// ---------------------------------------------------------------------------
// Model conversion
// ---------------------------------------------------------------------------

/**
 * Extract models for a specific provider from the models.dev response.
 * Filters to tool_call-capable models and excludes deprecated ones.
 * Sorts by release_date descending (newest first).
 */
export function getModelsForProvider(providerId: string, data: ModelsDevResponse): ModelDefinition[] {
  const providerData = data[providerId]
  if (!providerData?.models) return []

  const models: ModelDefinition[] = []

  for (const [, model] of Object.entries(providerData.models)) {
    // Skip models that don't support tool calls (we need them for agent use)
    if (!model.tool_call) continue
    // Skip deprecated models
    if (model.status === 'deprecated') continue

    models.push({
      id: model.id,
      name: model.name,
      contextWindow: model.limit?.context,
      maxOutputTokens: model.limit?.output,
      supportsToolCalls: model.tool_call,
      supportsReasoning: model.reasoning,
      cost: model.cost,
      family: model.family,
      releaseDate: model.release_date,
      status: model.status as ModelDefinition['status'],
    })
  }

  // Sort by release date descending (newest first), undated at end
  models.sort((a, b) => {
    if (!a.releaseDate && !b.releaseDate) return 0
    if (!a.releaseDate) return 1
    if (!b.releaseDate) return -1
    return b.releaseDate.localeCompare(a.releaseDate)
  })

  return models
}

// ---------------------------------------------------------------------------
// Populate registry from fetched data
// ---------------------------------------------------------------------------

function applyModels(data: ModelsDevResponse): void {
  populateModels((providerId) => getModelsForProvider(providerId, data))
}

// ---------------------------------------------------------------------------
// Background refresh
// ---------------------------------------------------------------------------

async function backgroundRefresh(): Promise<void> {
  const data = await fetchFromNetwork()
  if (data) {
    saveCache(data)
    applyModels(data)
    logger.info('Models refreshed from models.dev')
  }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/**
 * Initialize dynamic model lists from models.dev.
 *
 * Called once at startup. Loads from cache first for speed,
 * falls back to network, then stale cache, then leaves built-in fallbacks.
 * Starts a background refresh interval.
 */
export async function initializeModels(): Promise<void> {
  // 1. Try loading cache
  const cached = loadCache()

  if (cached && isCacheValid(cached)) {
    // Valid cache — use it immediately, refresh in background
    applyModels(cached.data)
    logger.info('Models loaded from cache')
    // Fire-and-forget background refresh
    backgroundRefresh().catch(() => {})
  } else {
    // Cache missing or stale — try network
    const data = await fetchFromNetwork()
    if (data) {
      saveCache(data)
      applyModels(data)
      logger.info('Models fetched from models.dev')
    } else if (cached) {
      // Network failed but we have stale cache — use it
      applyModels(cached.data)
      logger.info('Models loaded from stale cache (network unavailable)')
    } else {
      // No cache, no network — fallback models in registry remain as-is
      logger.warn('No models cache and network unavailable — using built-in fallbacks')
    }
  }

  // Start periodic background refresh
  const interval = setInterval(() => {
    backgroundRefresh().catch(() => {})
  }, REFRESH_INTERVAL_MS)
  interval.unref()
}
