/**
 * React Edit Benchmark — measures edit precision across edit formats × retry counts.
 *
 * Each fixture is a React source file with a single mutation injected.
 * The LLM is given the mutated file + a prompt describing the fix,
 * and must produce the correct edit using each format's instructions.
 *
 * Dimensions:
 *   - Format: whole, aider-replace, anthropic-replace, patch, hashline, xml-edit, xml-edit-trailing, xml-hybrid
 *   - Retries: 0 (single-shot), 1 (one retry with error feedback)
 *
 * Scenario IDs: {format}:r{retries}/{fixture-id}
 * Variants: {format}:r{retries}
 */

import type { ChatMessage } from '@magnitudedev/llm-core'
import { callModel } from '../../runner'
import { loadFixtures, type EditFixture } from './fixtures'
import { verify } from './verify'
import type { EditFormat } from './formats/types'
import { wholeFormat } from './formats/whole'
import { aiderReplaceFormat } from './formats/aider-replace'
import { anthropicReplaceFormat } from './formats/anthropic-replace'
import { patchFormat } from './formats/patch'
import { hashlineFormat } from './formats/hashline'
import { xmlEditFormat } from './formats/xml-edit'
import { xmlEditTrailingFormat } from './formats/xml-edit-trailing'
import { xmlHybridFormat } from './formats/xml-hybrid'
import { xmlHybridStrictFormat } from './formats/xml-hybrid-strict'
import { openaiPatchFormat } from './formats/openai-patch'
import { jsActFormat } from './formats/js-act'
import { xmlReplaceFormat } from './formats/xml-replace'

import type { RunnableEval, EvalVariant, Scenario, ScenarioResult, ModelSpec, Check } from '../../types'

// =============================================================================
// Format registry
// =============================================================================

const FORMATS: EditFormat[] = [
  wholeFormat,
  aiderReplaceFormat,
  anthropicReplaceFormat,
  patchFormat,
  hashlineFormat,
  xmlEditFormat,
  xmlEditTrailingFormat,
  xmlHybridFormat,
  xmlHybridStrictFormat,
  openaiPatchFormat,
  jsActFormat,
  xmlReplaceFormat,
]


const RETRY_COUNTS = [0, 1]

// =============================================================================
// Build scenarios from fixtures × formats × retries
// =============================================================================

function buildScenarios(fixtures: EditFixture[]): Scenario[] {
  const scenarios: Scenario[] = []

  for (const format of FORMATS) {
    for (const retries of RETRY_COUNTS) {
      for (const fixture of fixtures) {
        const filenames = Object.keys(fixture.inputFiles)
        if (filenames.length === 0) continue
        const filename = filenames[0]
        const inputContent = fixture.inputFiles[filename]

        const formattedFile = format.formatFile(filename, inputContent)

        const userMessage = [
          `File: ${filename}`,
          '```',
          formattedFile,
          '```',
          '',
          fixture.prompt,
        ].join('\n')

        const messages: ChatMessage[] = [
          { role: 'user', content: [userMessage] },
        ]

        const checks: Check[] = [{
          id: 'edit-correct',
          description: 'The edit produces the expected output file',
          evaluate() {
            return { passed: false, message: 'Not evaluated' }
          }
        }]

        scenarios.push({
          id: `${format.id}:r${retries}/${fixture.id}`,
          description: `[${format.id} r${retries}] ${fixture.metadata.mutationType} — ${fixture.metadata.difficulty}`,
          messages,
          checks,
        })
      }
    }
  }

  return scenarios
}

// =============================================================================
// Scenario execution
// =============================================================================

/** Parse scenario ID into format ID, retry count, and fixture ID */
function parseScenarioId(id: string): { formatId: string; retries: number; fixtureId: string } {
  // Format: {formatId}:r{retries}/{fixtureId}
  const slashIdx = id.indexOf('/')
  if (slashIdx === -1) throw new Error(`Invalid scenario ID: ${id}`)

  const prefix = id.slice(0, slashIdx)
  const fixtureId = id.slice(slashIdx + 1)

  const retryMatch = prefix.match(/^(.+):r(\d+)$/)
  if (!retryMatch) throw new Error(`Invalid scenario prefix: ${prefix}`)

  return {
    formatId: retryMatch[1],
    retries: parseInt(retryMatch[2], 10),
    fixtureId,
  }
}

function makeFail(scenarioId: string, message: string, rawResponse: string = ''): ScenarioResult {
  return {
    scenarioId,
    checks: { 'edit-correct': { passed: false, message } },
    passed: false,
    score: 0,
    rawResponse,
  }
}

async function executeScenario(
  scenario: Scenario,
  modelSpec: ModelSpec,
  fixtures: EditFixture[],
): Promise<ScenarioResult> {
  const { formatId, retries, fixtureId } = parseScenarioId(scenario.id)

  const format = FORMATS.find(f => f.id === formatId)
  if (!format) return makeFail(scenario.id, `Unknown format: ${formatId}`)

  const fixture = fixtures.find(f => f.id === fixtureId)
  if (!fixture) return makeFail(scenario.id, `Unknown fixture: ${fixtureId}`)

  const systemPrompt = [
    'You are a code editor. You will be given a file and an instruction to fix it.',
    '',
    format.systemInstructions(),
  ].join('\n')

  const filenames = Object.keys(fixture.inputFiles)
  const filename = filenames[0]
  const originalContent = fixture.inputFiles[filename]
  const expectedContent = fixture.expectedFiles[filename]

  // Conversation accumulates across retries
  const messages: ChatMessage[] = [...scenario.messages as ChatMessage[]]
  let lastRawResponse = ''

  for (let attempt = 0; attempt <= retries; attempt++) {
    // Call the model
    let rawResponse: string
    try {
      rawResponse = await callModel(systemPrompt, messages, modelSpec)
    } catch (error) {
      return makeFail(scenario.id, `Model call failed: ${error instanceof Error ? error.message : String(error)}`)
    }

    lastRawResponse = rawResponse

    // Try to apply the edit
    let editedContent: string
    let applyError: string | null = null
    try {
      editedContent = await format.applyResponse(rawResponse, originalContent)

    } catch (error) {
      applyError = error instanceof Error ? error.message : String(error)
      editedContent = ''
    }

    // If apply succeeded, verify
    if (!applyError) {
      const result = verify(expectedContent, editedContent)
      if (result.passed) {
        return {
          scenarioId: scenario.id,
          checks: { 'edit-correct': { passed: true } },
          passed: true,
          score: 1,
          rawResponse: lastRawResponse,
        }
      }

      // Verify failed — if we have retries left, feed back the error
      if (attempt < retries) {
        messages.push({ role: 'assistant', content: [rawResponse] })
        messages.push({
          role: 'user',
          content: [`The edit applied but produced incorrect output. ${result.error}\n\nPlease try again.`],
        })
        continue
      }

      // No retries left
      return {
        scenarioId: scenario.id,
        checks: {
          'edit-correct': { passed: false, message: result.error, snippet: result.diff?.slice(0, 500) },
        },
        passed: false,
        score: 0,
        rawResponse: lastRawResponse,
      }
    }

    // Apply failed — if we have retries left, feed back the error
    if (attempt < retries) {
      messages.push({ role: 'assistant', content: [rawResponse] })
      messages.push({
        role: 'user',
        content: [`Your edit could not be applied:\n${applyError}\n\nPlease try again with a corrected edit.`],
      })
      continue
    }

    // No retries left
    return makeFail(scenario.id, `Apply failed: ${applyError}`, lastRawResponse)
  }

  // Should not reach here, but satisfy TypeScript
  return makeFail(scenario.id, 'Unexpected: exhausted retry loop', lastRawResponse)
}

// =============================================================================
// Eval export
// =============================================================================

const fixtures = loadFixtures()
const scenarios = buildScenarios(fixtures)

const variants: EvalVariant[] = []
for (const format of FORMATS) {
  for (const retries of RETRY_COUNTS) {
    const variantId = `${format.id}:r${retries}`
    variants.push({
      id: variantId,
      label: retries === 0 ? format.id : `${format.id} +${retries}retry`,
      count: fixtures.length,
    })
  }
}

export const reactEditEval: RunnableEval = {
  id: 'react-edit',
  name: 'React Edit Benchmark',
  description: `Measures edit precision across ${FORMATS.length} formats × ${RETRY_COUNTS.length} retry configs on ${fixtures.length} fixtures (${scenarios.length} total scenarios)`,
  scenarios,
  variants,
  defaultConcurrency: 16,

  async runScenario(scenario: Scenario, modelSpec: ModelSpec): Promise<ScenarioResult> {
    return executeScenario(scenario, modelSpec, fixtures)
  },
}
