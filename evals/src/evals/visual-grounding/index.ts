/**
 * Visual Grounding Eval
 *
 * Tests LLM ability to identify and localize visual targets in browser screenshots.
 * Renders 4 labeled circles at known positions, asks each model for click coordinates,
 * and scores accuracy by Euclidean distance from circle centers.
 *
 * Two viewport variants: 1024x768 and 1280x720.
 */

import { join } from 'path'
import { writeFileSync, mkdirSync } from 'fs'
import { tmpdir } from 'os'
import { Image as BamlImage } from '@boundaryml/baml'
import { BrowserProvider, WebHarness } from '@magnitudedev/browser-harness'
import type { RunnableEval, Scenario, ScenarioResult, ModelSpec, Check, CheckResult, EvalVariant, ChatMessage } from '../../types'
import { callModel } from '../../runner'
import { VIEWPORTS, VIEWPORT_IDS, type ViewportConfig } from './targets'
import { SYSTEM_PROMPT, USER_PROMPT } from './prompt'
import { parseCoordinates, scoreCircle } from './scoring'
import { generateCirclesHtml } from './circles'
import { VISUAL_GROUNDING_MODELS } from './models'

// ─── Screenshot Cache ───────────────────────────────────────────────
// One screenshot per viewport, lazily created
const screenshotCache = new Map<string, Promise<string>>()

async function createScreenshot(viewportId: string): Promise<string> {
  const config = VIEWPORTS[viewportId]
  if (!config) throw new Error(`Unknown viewport: ${viewportId}`)

  // Write HTML to temp file
  const html = generateCirclesHtml(config)
  const tmpDir = join(tmpdir(), 'magnitude-visual-grounding')
  mkdirSync(tmpDir, { recursive: true })
  const htmlPath = join(tmpDir, `circles-${viewportId}.html`)
  writeFileSync(htmlPath, html, 'utf-8')

  // Launch headless browser at target viewport
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

  // Wait for rendering
  await harness.page.waitForTimeout(500)

  const screenshot = await harness.screenshot()
  const base64 = await screenshot.toBase64()

  await harness.stop()
  await context.close()

  return base64
}

function getScreenshot(viewportId: string): Promise<string> {
  let promise = screenshotCache.get(viewportId)
  if (!promise) {
    promise = createScreenshot(viewportId)
    screenshotCache.set(viewportId, promise)
  }
  return promise
}

// ─── Checks ─────────────────────────────────────────────────────────
// Two checks per circle:
//   circle-X-distance — raw pixel distance from center (primary accuracy metric)
//   circle-X-hit      — did the click land inside the circle?

const MAX_SCORE_DISTANCE = 150 // distance at which score reaches 0

function buildChecks(viewportId: string): Check[] {
  const config = VIEWPORTS[viewportId]
  const checks: Check[] = []

  for (const target of config.targets) {
    const label = target.label.toLowerCase()

    // Check 1: Raw distance from center
    checks.push({
      id: `circle-${label}-distance`,
      description: `Pixel distance from center of circle ${target.label}`,
      evaluate: (rawResponse: string): CheckResult => {
        const parsed = parseCoordinates(rawResponse)
        const predicted = parsed.find(p => p.label === target.label)
        const result = scoreCircle(predicted, target)

        if (result.distance === null) {
          return { passed: false, score: 0, message: `Circle ${target.label} not found in response` }
        }

        // Score: 1.0 at 0px, linear decay to 0.0 at MAX_SCORE_DISTANCE
        const score = Math.max(0, 1.0 - result.distance / MAX_SCORE_DISTANCE)

        return {
          passed: result.distance <= MAX_SCORE_DISTANCE,
          score: Math.round(score * 1000) / 1000,
          message: `${result.distance}px from center → predicted (${result.predictedX}, ${result.predictedY}) vs actual (${result.actualX}, ${result.actualY})`,
        }
      },
    })

    // Check 2: Inside the circle?
    checks.push({
      id: `circle-${label}-hit`,
      description: `Click landed inside circle ${target.label} (radius ${target.radius}px)`,
      evaluate: (rawResponse: string): CheckResult => {
        const parsed = parseCoordinates(rawResponse)
        const predicted = parsed.find(p => p.label === target.label)
        const result = scoreCircle(predicted, target)

        if (result.distance === null) {
          return { passed: false, score: 0, message: `Circle ${target.label} not found in response` }
        }

        return {
          passed: result.withinRadius,
          score: result.withinRadius ? 1 : 0,
          message: result.withinRadius
            ? `Hit! ${result.distance}px from center (radius: ${target.radius}px)`
            : `Miss — ${result.distance}px from center (radius: ${target.radius}px)`,
        }
      },
    })
  }

  return checks
}

// ─── Scenarios ──────────────────────────────────────────────────────
const scenarios: Scenario[] = VIEWPORT_IDS.map(viewportId => ({
  id: `${viewportId}/identify-circles`,
  description: `Identify 4 circle centers at ${viewportId}`,
  messages: [], // built dynamically with screenshot
  checks: buildChecks(viewportId),
}))

const variants: EvalVariant[] = VIEWPORT_IDS.map(viewportId => ({
  id: viewportId,
  label: viewportId,
  count: 1,
}))

// ─── Scenario Execution ─────────────────────────────────────────────
async function executeScenario(
  scenario: Scenario,
  modelSpec: ModelSpec,
): Promise<ScenarioResult> {
  const viewportId = scenario.id.split('/')[0]

  // Get screenshot
  let base64: string
  try {
    base64 = await getScreenshot(viewportId)
  } catch (error) {
    return failResult(scenario, `Browser setup failed: ${error instanceof Error ? error.message : String(error)}`)
  }

  // Build multimodal message
  const bamlImage = BamlImage.fromBase64('image/png', base64)
  const messages: ChatMessage[] = [
    {
      role: 'user',
      content: [USER_PROMPT, bamlImage],
    },
  ]

  // Call model
  let rawResponse: string
  try {
    rawResponse = await callModel(SYSTEM_PROMPT, messages, modelSpec)
  } catch (error) {
    return failResult(scenario, `Model call failed: ${error instanceof Error ? error.message : String(error)}`)
  }

  // Evaluate checks
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

// ─── Export ──────────────────────────────────────────────────────────
export const visualGroundingEval: RunnableEval = {
  id: 'visual-grounding',
  name: 'Visual Grounding Accuracy',
  description: 'Tests LLM ability to localize visual targets in browser screenshots',
  scenarios,
  variants,
  defaultConcurrency: 2,
  defaultModels: VISUAL_GROUNDING_MODELS,

  async runScenario(scenario: Scenario, modelSpec: ModelSpec): Promise<ScenarioResult> {
    return executeScenario(scenario, modelSpec)
  },
}
