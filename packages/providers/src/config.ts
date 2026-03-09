/**
 * Auth and config persistence for multi-provider support.
 *
 * - ~/.magnitude/auth.json  — credential storage (mode 0o600)
 * - ~/.magnitude/config.json — active provider/model, non-secret preferences
 */

import * as path from 'path'
import * as fs from 'fs'
import type { AuthInfo, MagnitudeConfig } from './types'

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

const DATA_DIR = path.join(process.env.HOME ?? '~', '.magnitude')
const AUTH_PATH = path.join(DATA_DIR, 'auth.json')
const CONFIG_PATH = path.join(DATA_DIR, 'config.json')

function ensureDir() {
  if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true })
  }
}

// ---------------------------------------------------------------------------
// Auth storage  (~/.magnitude/auth.json, mode 0o600)
// ---------------------------------------------------------------------------

/** Read all stored auth entries, keyed by provider ID */
export function loadAuth(): Record<string, AuthInfo> {
  try {
    const raw = fs.readFileSync(AUTH_PATH, 'utf-8')
    const data = JSON.parse(raw)
    if (typeof data !== 'object' || data === null) return {}
    // Validate each entry has a known type
    const result: Record<string, AuthInfo> = {}
    for (const [key, value] of Object.entries(data)) {
      if (isValidAuthInfo(value)) {
        result[key] = value as AuthInfo
      }
    }
    return result
  } catch {
    return {}
  }
}

/** Get auth info for a specific provider */
export function getAuth(providerId: string): AuthInfo | undefined {
  return loadAuth()[providerId]
}

/** Store auth info for a provider (creates file with 0o600 permissions) */
export function setAuth(providerId: string, info: AuthInfo): void {
  ensureDir()
  const data = loadAuth()
  data[providerId] = info
  writeSecure(AUTH_PATH, data)
}

/** Remove auth info for a provider */
export function removeAuth(providerId: string): void {
  const data = loadAuth()
  delete data[providerId]
  if (Object.keys(data).length === 0) {
    // Remove the file entirely if empty
    try { fs.unlinkSync(AUTH_PATH) } catch { /* ignore */ }
  } else {
    ensureDir()
    writeSecure(AUTH_PATH, data)
  }
}

// ---------------------------------------------------------------------------
// Config storage  (~/.magnitude/config.json)
// ---------------------------------------------------------------------------

const DEFAULT_CONFIG: MagnitudeConfig = {
  primaryModel: null,
  secondaryModel: null,
  browserModel: null,
}

/** Load non-secret config (provider/model selection, options) */
export function loadConfig(): MagnitudeConfig {
  try {
    const raw = fs.readFileSync(CONFIG_PATH, 'utf-8')
    const data = JSON.parse(raw)
    return {
      primaryModel: data.primaryModel ?? null,
      secondaryModel: data.secondaryModel ?? null,
      browserModel: data.browserModel ?? null,
      providerOptions: data.providerOptions ?? undefined,
      setupComplete: data.setupComplete ?? false,
      machineId: data.machineId ?? undefined,
      telemetry: data.telemetry ?? undefined,
      memory: data.memory ?? undefined,
    }
  } catch {
    return { ...DEFAULT_CONFIG }
  }
}

/** Save config */
export function saveConfig(config: MagnitudeConfig): void {
  ensureDir()
  fs.writeFileSync(
    CONFIG_PATH,
    JSON.stringify(config, null, 2) + '\n',
    'utf-8'
  )
}

/** Convenience: update primary model selection */
export function setPrimarySelection(providerId: string, modelId: string): void {
  const config = loadConfig()
  config.primaryModel = { providerId, modelId }
  saveConfig(config)
}

/** Convenience: update browser model selection */
export function setBrowserSelection(providerId: string, modelId: string): void {
  const config = loadConfig()
  config.browserModel = { providerId, modelId }
  saveConfig(config)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Write JSON with restrictive permissions (0o600 — owner read/write only) */
function writeSecure(filePath: string, data: unknown): void {
  const content = JSON.stringify(data, null, 2) + '\n'
  fs.writeFileSync(filePath, content, { mode: 0o600 })
}

/** Validate that a value looks like a valid AuthInfo entry */
function isValidAuthInfo(value: unknown): boolean {
  if (typeof value !== 'object' || value === null) return false
  const v = value as Record<string, unknown>
  switch (v.type) {
    case 'api':
      return typeof v.key === 'string'
    case 'oauth':
      return typeof v.accessToken === 'string' && typeof v.refreshToken === 'string' && typeof v.expiresAt === 'number'
    case 'aws':
      return true
    case 'gcp':
      return typeof v.credentialsPath === 'string'
    default:
      return false
  }
}
