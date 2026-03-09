/**
 * Scenario Loader — reads scenario definitions from the scenarios/ directory.
 */

import { readdirSync, readFileSync } from 'fs'
import { join, dirname } from 'path'

export interface ScenarioDef {
  id: string
  description: string
  taskPrompt: string
  verifyCommand: string
  language: 'node' | 'python'
  protectedFiles: string[]
  /** Absolute path to the scenario directory (contains Dockerfile, project/) */
  scenarioDir: string
}

const SCENARIOS_DIR = dirname(import.meta.path)

export function loadScenarios(): ScenarioDef[] {
  const entries = readdirSync(SCENARIOS_DIR, { withFileTypes: true })
    .filter(e => e.isDirectory())

  const scenarios: ScenarioDef[] = []

  for (const entry of entries) {
    const scenarioDir = join(SCENARIOS_DIR, entry.name)
    const jsonPath = join(scenarioDir, 'scenario.json')

    try {
      const raw = JSON.parse(readFileSync(jsonPath, 'utf-8'))
      scenarios.push({
        id: raw.id,
        description: raw.description,
        taskPrompt: raw.taskPrompt,
        verifyCommand: raw.verifyCommand,
        language: raw.language,
        protectedFiles: raw.protectedFiles ?? [],
        scenarioDir,
      })
    } catch {
      // Skip directories without scenario.json
    }
  }

  return scenarios
}
