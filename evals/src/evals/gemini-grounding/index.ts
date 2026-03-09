/**
 * Gemini Visual Grounding Eval
 *
 * Tests Gemini's bounding-box coordinate system (1000x1000 normalized grid)
 * and verifies that converting those coordinates back to the actual viewport
 * produces accurate click targets.
 *
 * This eval:
 * 1. Takes a screenshot at 1024x768 (standard browser viewport)
 * 2. Asks Gemini to report coordinates — Gemini returns them in 0-1000 space
 * 3. Converts the 1000-space coordinates to 1024x768 for scoring
 * 4. Scores accuracy against known circle positions
 *
 * This validates the virtualScreenDimensions approach used in browser-service.ts.
 */

import { join } from 'path'
import { writeFileSync, mkdirSync } from 'fs'
import { tmpdir } from 'os'
import { Image as BamlImage } from '@boundaryml/baml'
import { BrowserProvider, WebHarness } from '@magnitudedev/browser-harness'
import type { RunnableEval, Scenario, ScenarioResult, ModelSpec, Check, CheckResult, EvalVariant, ChatMessage } from '../../types'
import { callModel } from '../../runner'
import { VIEWPORTS, type ViewportConfig } from '../visual-grounding/targets'
import { generateCirclesHtml } from '../visual-grounding/circles'
import { parseCoordinates, scoreCircle } from './scoring'

// ─── Constants ────────────────────────────────────────────────────
const VIEWPORT_ID = '1024x768'
const GEMINI_GRID_SIZE = 1000

// ─── Prompts ──────────────────────────────────────────────────────

const SYSTEM_PROMPT = `You are a visual analysis assistant. You are looking at a screenshot of a web page. Your task is to identify visual elements and report their coordinates.

IMPORTANT: Report all coordinates in a normalized coordinate space where both axes range from 0 to 1000, regardless of the actual image dimensions. The top-left corner is (0, 0) and the bottom-right corner is (1000, 1000).`

const USER_PROMPT = `Look at this screenshot carefully. There are four colored circles on the page, each labeled with a letter (A, B, C, D).

For each circle, identify the center point and report its coordinates in the 0-1000 normalized coordinate space.

Report your answer in this exact XML format:

<coordinates>
  <circle label="A" x="___" y="___" />
  <circle label="B" x="___" y="___" />
  <circle label="C" x="___" y="___" />
  <circle label="D" x="___" y="___" />
</coordinates>

Replace ___ with integer coordinates in the 0-1000 range. Be as precise as possible. Do not explain your reasoning - just output the coordinates in the XML format above.`

// ─── Screenshot Cache ─────────────────────────────────────────────
let screenshotPromise: Promise<string> | null = null

async function createScreenshot(): Promise<string> {
  const config = VIEWPORTS[VIEWPORT_ID]
  if (!config) throw new Error(`Unknown viewport: ${VIEWPORT_ID}`)

  const html = generateCirclesHtml(config)
  const tmpDir = join(tmpdir(), 'magnitude-gemini-grounding')
  mkdirSync(tmpDir, { recursive: true })
  const htmlPath = join(tmpDir, `circles-${VIEWPORT_ID}.html`)
  writeFileSync(htmlPath, html, 'utf-8')

  const provider = BrowserProvider.getInstance()
  const context = await provider.newContext({
    launchOptions: { headless: true },
    contextOptions: { viewport: { width: config.width, height: config.height } },
  })
  const harness = new WebHarness(context, {
    virtualScreenDimensions: { width: config.width, height: config.height },
  })
  await harness.start()
  await harness.navigate(`file://${htmlPath}`)
  await harness.page.waitForTimeout(500)

  const screenshot = await harness.screenshot()
  const base64 = await screenshot.toBase64()

  await harness.stop()
  await context.close()

  return base64
}

function getScreenshot(): Promise<string> {
  if (!screenshotPromise) {
    screenshotPromise = createScreenshot()
  }
  return screenshotPromise
}

// ─── Coordinate Conversion ────────────────────────────────────────

/**
 * Convert coordinates from Gemini's 1000x1000 space to actual viewport pixels.
 * This mirrors what transformCoordinates() does in the harness.
 */
function geminiToViewport(x: number, y: number, config: ViewportConfig): { x: number; y: number } {
  return {
    x: Math.round(x * (config.width / GEMINI_GRID_SIZE)),
    y: Math.round(y * (config.height / GEMINI_GRID_SIZE)),
  }
}

// ─── Checks ───────────────────────────────────────────────────────

const MAX_SCORE_DISTANCE = 150

function buildChecks(): Check[] {
  const config = VIEWPORTS[VIEWPORT_ID]
  const checks: Check[] = []

  for (const target of config.targets) {
    const label = target.label.toLowerCase()

    // Check 1: Distance after converting from 1000-space to viewport-space
    checks.push({
      id: `circle-${label}-distance`,
      description: `Pixel distance from center of circle ${target.label} (after 1000→viewport conversion)`,
      evaluate: (rawResponse: string): CheckResult => {
        const parsed = parseCoordinates(rawResponse)
        const predicted = parsed.find(p => p.label === target.label)

        if (!predicted) {
          return { passed: false, score: 0, message: `Circle ${target.label} not found in response` }
        }

        // Convert from Gemini 1000-space to viewport pixels
        const converted = geminiToViewport(predicted.x, predicted.y, config)
        const result = scoreCircle({ label: predicted.label, x: converted.x, y: converted.y }, target)

        if (result.distance === null) {
          return { passed: false, score: 0, message: `Circle ${target.label} scoring failed` }
        }

        const score = Math.max(0, 1.0 - result.distance / MAX_SCORE_DISTANCE)

        return {
          passed: result.distance <= MAX_SCORE_DISTANCE,
          score: Math.round(score * 1000) / 1000,
          message: `Gemini (${predicted.x}, ${predicted.y}) → viewport (${converted.x}, ${converted.y}) | ${result.distance}px from actual (${result.actualX}, ${result.actualY})`,
        }
      },
    })

    // Check 2: Hit test after conversion
    checks.push({
      id: `circle-${label}-hit`,
      description: `Click would land inside circle ${target.label} after coordinate conversion`,
      evaluate: (rawResponse: string): CheckResult => {
        const parsed = parseCoordinates(rawResponse)
        const predicted = parsed.find(p => p.label === target.label)

        if (!predicted) {
          return { passed: false, score: 0, message: `Circle ${target.label} not found in response` }
        }

        const converted = geminiToViewport(predicted.x, predicted.y, config)
        const result = scoreCircle({ label: predicted.label, x: converted.x, y: converted.y }, target)

        if (result.distance === null) {
          return { passed: false, score: 0, message: `Circle ${target.label} scoring failed` }
        }

        return {
          passed: result.withinRadius,
          score: result.withinRadius ? 1 : 0,
          message: result.withinRadius
            ? `Hit! Gemini (${predicted.x}, ${predicted.y}) → viewport (${converted.x}, ${converted.y}) | ${result.distance}px from center (radius: ${target.radius}px)`
            : `Miss — Gemini (${predicted.x}, ${predicted.y}) → viewport (${converted.x}, ${converted.y}) | ${result.distance}px from center (radius: ${target.radius}px)`,
        }
      },
    })

    // Check 3: Raw coordinates — are they in the expected 0-1000 range?
    checks.push({
      id: `circle-${label}-range`,
      description: `Circle ${target.label} coordinates are in 0-1000 range`,
      evaluate: (rawResponse: string): CheckResult => {
        const parsed = parseCoordinates(rawResponse)
        const predicted = parsed.find(p => p.label === target.label)

        if (!predicted) {
          return { passed: false, score: 0, message: `Circle ${target.label} not found in response` }
        }

        const inRange = predicted.x >= 0 && predicted.x <= 1000 && predicted.y >= 0 && predicted.y <= 1000
        return {
          passed: inRange,
          score: inRange ? 1 : 0,
          message: inRange
            ? `In range: (${predicted.x}, ${predicted.y})`
            : `OUT OF RANGE: (${predicted.x}, ${predicted.y}) — model may be returning pixel coordinates instead of 1000-space`,
        }
      },
    })
  }

  return checks
}

// ─── Scenarios ────────────────────────────────────────────────────

const scenarios: Scenario[] = [{
  id: `${VIEWPORT_ID}/gemini-grounding`,
  description: `Identify 4 circle centers using Gemini 1000-space coordinates at ${VIEWPORT_ID}`,
  messages: [],
  checks: buildChecks(),
}]

const variants: EvalVariant[] = [{
  id: VIEWPORT_ID,
  label: VIEWPORT_ID,
  count: 1,
}]

// ─── Models ───────────────────────────────────────────────────────

const GEMINI_MODELS: ModelSpec[] = [
  { provider: 'google', model: 'gemini-3.1-pro-preview', label: 'google:gemini-3.1-pro-preview' },
  { provider: 'google', model: 'gemini-3-pro-preview', label: 'google:gemini-3-pro-preview' },
  { provider: 'google', model: 'gemini-3-flash-preview', label: 'google:gemini-3-flash-preview' },
  { provider: 'google', model: 'gemini-2.5-flash-lite', label: 'google:gemini-2.5-flash-lite' },
]

// ─── Execution ────────────────────────────────────────────────────

async function executeScenario(
  scenario: Scenario,
  modelSpec: ModelSpec,
): Promise<ScenarioResult> {
  let base64: string
  try {
    base64 = await getScreenshot()
  } catch (error) {
    return failResult(scenario, `Browser setup failed: ${error instanceof Error ? error.message : String(error)}`)
  }

  const bamlImage = BamlImage.fromBase64('image/png', base64)
  const messages: ChatMessage[] = [
    {
      role: 'user',
      content: [USER_PROMPT, bamlImage],
    },
  ]

  let rawResponse: string
  try {
    rawResponse = await callModel(SYSTEM_PROMPT, messages, modelSpec)
  } catch (error) {
    return failResult(scenario, `Model call failed: ${error instanceof Error ? error.message : String(error)}`)
  }

  const checks: Record<string, CheckResult> = {}
  let allPassed = true
  for (const check of scenario.checks) {
    const result = check.evaluate(rawResponse, {} as any)
    checks[check.id] = result
    if (!result.passed) allPassed = false
  }

  const checkResults = Object.values(checks)
  const score = checkResults.length > 0
    ? checkResults.reduce((sum, c) => sum + (c.score ?? (c.passed ? 1 : 0)), 0) / checkResults.length
    : 0

  return {
    scenarioId: scenario.id,
    checks,
    passed: allPassed,
    score,
    rawResponse,
  }
}

function failResult(scenario: Scenario, message: string): ScenarioResult {
  return {
    scenarioId: scenario.id,
    checks: Object.fromEntries(
      scenario.checks.map(c => [c.id, { passed: false, score: 0, message }])
    ),
    passed: false,
    score: 0,
    rawResponse: '',
  }
}

// ─── Export ───────────────────────────────────────────────────────

export const geminiGroundingEval: RunnableEval = {
  id: 'gemini-grounding',
  name: 'Gemini Visual Grounding (1000-space)',
  description: 'Tests Gemini coordinate conversion from 1000x1000 normalized grid to viewport pixels',
  scenarios,
  variants,
  defaultConcurrency: 2,
  defaultModels: GEMINI_MODELS,

  async runScenario(scenario: Scenario, modelSpec: ModelSpec): Promise<ScenarioResult> {
    return executeScenario(scenario, modelSpec)
  },
}
