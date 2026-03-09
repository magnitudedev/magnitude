/**
 * Anonymous Machine ID — generates and persists a random CUID2 per installation.
 *
 * Stored in ~/.magnitude/config.json alongside other config.
 * Used as PostHog distinctId for anonymous telemetry.
 */

import * as fs from 'fs'
import * as path from 'path'
import { createId } from '@magnitudedev/generate-id'

const DATA_DIR = path.join(process.env.HOME ?? '~', '.magnitude')
const CONFIG_PATH = path.join(DATA_DIR, 'config.json')

/**
 * Get or create an anonymous machine ID.
 * Reads from ~/.magnitude/config.json, generates a new CUID2 if not present.
 */
export function getOrCreateMachineId(): string {
  let config: Record<string, unknown> = {}
  try {
    const raw = fs.readFileSync(CONFIG_PATH, 'utf-8')
    config = JSON.parse(raw)
  } catch {
    // File doesn't exist or is invalid — will create
  }

  if (typeof config.machineId === 'string' && config.machineId.length > 0) {
    return config.machineId
  }

  const machineId = createId()
  config.machineId = machineId

  if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true })
  }
  fs.writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2) + '\n', 'utf-8')

  return machineId
}
