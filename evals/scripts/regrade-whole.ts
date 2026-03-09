/**
 * Retroactively regrade whole-file format results using improved response extraction.
 *
 * Reads the raw LLM responses from the results JSON, re-applies the updated
 * extractFileContent logic, re-verifies against expected fixtures, and writes
 * updated result files.
 */
import { readFileSync, writeFileSync, readdirSync } from 'fs'
import { join } from 'path'
import { verify } from '../src/evals/react-edit/verify'
import { loadFixtures } from '../src/evals/react-edit/fixtures'

// --- Inline the extractFileContent logic (must match whole.ts) ---

function buildAnchor(originalContent: string): string | null {
  const lines = originalContent.split('\n')
  let anchor = ''
  for (const line of lines) {
    anchor += (anchor ? '\n' : '') + line
    if (anchor.length >= 50) return anchor
  }
  return anchor.length > 0 ? anchor : null
}

const FENCE = '\`\`\`'
const FENCE_RE = new RegExp(FENCE + '\\w*\\n([\\s\\S]*?)' + FENCE, 'g')

function extractFileContent(raw: string, originalContent: string): string {
  const content = raw.trim()
  const anchor = buildAnchor(originalContent)

  if (anchor && content.startsWith(anchor)) {
    return content
  }

  if (anchor) {
    const idx = content.indexOf(anchor)
    if (idx > 0) {
      const beforeAnchor = content.slice(0, idx)
      const lastFenceOpen = beforeAnchor.lastIndexOf(FENCE)
      if (lastFenceOpen >= 0) {
        const afterAnchor = content.indexOf(FENCE, idx)
        if (afterAnchor > idx) {
          return content.slice(idx, afterAnchor)
        }
      }
      return content.slice(idx)
    }
  }

  const fenceMatches = [...content.matchAll(FENCE_RE)]
  if (fenceMatches.length > 0) {
    const best = fenceMatches.reduce((a, b) => a[1].length > b[1].length ? a : b)
    return best[1]
  }

  return content
}

// --- Main ---

const RESULTS_DIR = join(import.meta.dir, '../results/2026-03-01T07-56-21')

const fixtures = loadFixtures()
const fixtureMap = new Map(fixtures.map(f => [f.id, f]))

const jsonFiles = readdirSync(RESULTS_DIR).filter(f => f.endsWith('.json'))

let totalChanged = 0
let totalScenarios = 0

for (const jsonFile of jsonFiles) {
  const filePath = join(RESULTS_DIR, jsonFile)
  const data = JSON.parse(readFileSync(filePath, 'utf8'))
  const modelKey = `${data.provider}:${data.model}`

  let changed = 0

  for (const scenario of data.scenarios) {
    // Only regrade whole format scenarios
    if (!scenario.scenarioId.startsWith('whole:r0/') && !scenario.scenarioId.startsWith('whole:r1/')) continue

    totalScenarios++

    // Skip if already passed
    const alreadyPassed = Object.values(scenario.checks as Record<string, { passed: boolean }>).every(c => c.passed)
    if (alreadyPassed) continue

    // Skip if no raw response (rate limit failures etc)
    if (!scenario.rawResponse || scenario.rawResponse.length === 0) continue

    // Extract fixture ID
    const slashIdx = scenario.scenarioId.indexOf('/')
    const fixtureId = scenario.scenarioId.slice(slashIdx + 1)
    const fixture = fixtureMap.get(fixtureId)
    if (!fixture) continue

    const filenames = Object.keys(fixture.inputFiles)
    const filename = filenames[0]
    const originalContent = fixture.inputFiles[filename]
    const expectedContent = fixture.expectedFiles[filename]

    // Re-apply with improved extraction
    const editedContent = extractFileContent(scenario.rawResponse, originalContent)

    // Re-verify
    const result = verify(expectedContent, editedContent)

    if (result.passed) {
      // Upgrade to pass
      scenario.checks = { 'edit-correct': { passed: true } }
      scenario.passed = true
      scenario.score = 1
      changed++
    } else {
      // Still fails — update the error message in case it changed
      scenario.checks = {
        'edit-correct': {
          passed: false,
          message: result.error,
          snippet: result.diff?.slice(0, 500),
        }
      }
    }
  }

  if (changed > 0) {
    // Recalculate passedCount
    data.passedCount = data.scenarios.filter((s: { passed: boolean }) => s.passed).length

    writeFileSync(filePath, JSON.stringify(data, null, 2))
    console.log(`${modelKey}: ${changed} scenarios upgraded to PASS`)
    totalChanged += changed
  } else {
    console.log(`${modelKey}: no changes`)
  }
}

console.log(`\nTotal: ${totalChanged} scenarios regraded across ${totalScenarios} whole-format scenarios`)
