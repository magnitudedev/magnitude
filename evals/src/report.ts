/**
 * Report formatting for eval results — progressive and summary output
 */

import ansis from 'ansis'
import type { Scenario, ScenarioResult, EvalRunResult } from './types'
import { chatMessageToDisplayText } from './message-content'

const PASS = ansis.green('✓')
const FAIL = ansis.red('✗')
const DIVIDER = ansis.dim('─'.repeat(72))
const THIN_DIVIDER = ansis.dim('╌'.repeat(72))

// =============================================================================
// Progressive output — called as results arrive
// =============================================================================

/**
 * Print model header when starting a new model
 */
export function printModelHeader(label: string): void {
  console.log()
  console.log(ansis.bold.underline(label))
  console.log()
}

/**
 * Print scenario header with the conversation messages
 */
export function printScenarioHeader(scenario: Scenario, index: number, total: number): void {
  console.log(DIVIDER)
  console.log(ansis.bold(`  Scenario ${index + 1}/${total}: ${scenario.id}`))
  console.log(ansis.dim(`  ${scenario.description}`))
  console.log()

  // Show the conversation messages (abbreviated)
  for (const msg of scenario.messages) {
    const role = msg.role === 'user' ? ansis.blue.bold('USER') : ansis.green.bold('ASST')
    const content = abbreviateMessage(chatMessageToDisplayText(msg), 200)
    const lines = content.split('\n')
    console.log(`  ${role}  ${lines[0]}`)
    for (let i = 1; i < lines.length; i++) {
      console.log(`        ${lines[i]}`)
    }
  }
  console.log()
}

/**
 * Print the LLM's raw response
 */
export function printResponse(rawResponse: string): void {
  console.log(ansis.bold('  Response:'))
  console.log()
  if (!rawResponse) {
    console.log(ansis.dim('    (empty response)'))
  } else {
    // Show full response, indented
    const lines = rawResponse.split('\n')
    const maxLines = 60
    const shown = lines.slice(0, maxLines)
    for (const line of shown) {
      console.log(ansis.dim('    │ ') + line)
    }
    if (lines.length > maxLines) {
      console.log(ansis.dim(`    │ ... (${lines.length - maxLines} more lines)`))
    }
  }
  console.log()
}

/**
 * Print check results for a scenario
 */
export function printCheckResults(result: ScenarioResult): void {
  const icon = result.passed ? PASS : FAIL
  const status = result.passed ? ansis.green('PASSED') : ansis.red('FAILED')
  console.log(`  ${icon} ${status}`)

  for (const [checkId, check] of Object.entries(result.checks)) {
    const checkIcon = check.passed ? PASS : FAIL
    const resolvedScore = check.score ?? (check.passed ? 1 : 0)
    const showScore = resolvedScore !== 0 && resolvedScore !== 1
    const scoreSuffix = showScore ? ansis.dim(` (${resolvedScore.toFixed(2)})`) : ''

    if (check.passed) {
      console.log(ansis.dim(`    ${checkIcon} ${checkId}${scoreSuffix}`))
    } else {
      console.log(ansis.red(`    ${checkIcon} ${checkId}: ${check.message ?? 'failed'}${scoreSuffix}`))
      if (check.snippet) {
        const lines = check.snippet.split('\n').slice(0, 5)
        for (const line of lines) {
          console.log(ansis.dim(`       ${line}`))
        }
        if (check.snippet.split('\n').length > 5) {
          console.log(ansis.dim('       ...'))
        }
      }
    }
  }
  console.log()
}

/**
 * Print a complete scenario result (header + response + checks)
 */
export function printScenarioResult(
  scenario: Scenario,
  result: ScenarioResult,
  index: number,
  total: number
): void {
  printScenarioHeader(scenario, index, total)
  printResponse(result.rawResponse)
  printCheckResults(result)
}

// =============================================================================
// Summary — printed after all scenarios complete
// =============================================================================

/**
 * Print a summary for a model's eval run
 */
export function printModelSummary(result: EvalRunResult): void {
  console.log(DIVIDER)
  const pct = result.totalCount > 0 ? Math.round((result.passedCount / result.totalCount) * 100) : 0
  const status = result.passedCount === result.totalCount
    ? ansis.green.bold(`${result.passedCount}/${result.totalCount} (${pct}%)`)
    : ansis.red.bold(`${result.passedCount}/${result.totalCount} (${pct}%)`)

  console.log()
  console.log(`  ${ansis.bold('Summary:')} ${status}  ${ansis.bold('Score:')} ${result.averageScore.toFixed(2)}`)
  console.log()
}

/**
 * Print a final combined report (for multi-model runs)
 */
export function printReport(results: EvalRunResult[]): void {
  console.log()
  console.log(ansis.bold.underline('Final Results'))
  console.log()

  // Detect viewport variants from scenario IDs
  const viewports = new Set<string>()
  for (const run of results) {
    for (const sr of run.scenarios) {
      const slash = sr.scenarioId.indexOf('/')
      if (slash !== -1) viewports.add(sr.scenarioId.slice(0, slash))
    }
  }
  const vpList = [...viewports].sort()

  if (vpList.length > 0) {
    // Per-viewport header
    const vpHeaders = vpList.map(v => v.padStart(14)).join('')
    console.log(`  ${ansis.bold('Model'.padEnd(44))}${vpList.map(v => ansis.bold(v.padStart(14))).join('')}  ${ansis.bold('Overall')}`)
    console.log(`  ${'─'.repeat(44)}${vpList.map(() => '──────────────').join('')}${'─────────'}`)

    for (const run of results) {
      const modelLabel = `${run.provider}:${run.model}`
      const vpCells = vpList.map(vp => {
        const sr = run.scenarios.find(s => s.scenarioId.startsWith(vp + '/'))
        if (!sr) return '—'.padStart(14)

        // Count hits (circle-X-hit checks that passed)
        let hits = 0
        let hitTotal = 0
        for (const [checkId, check] of Object.entries(sr.checks)) {
          if (checkId.endsWith('-hit')) {
            hitTotal++
            if (check.passed) hits++
          }
        }
        const hitStr = `${hits}/${hitTotal} hit`
        return hitStr.padStart(14)
      })

      const overall = run.averageScore.toFixed(2)
      const overallColor = run.averageScore >= 0.9 ? ansis.green(overall) : run.averageScore >= 0.5 ? ansis.yellow(overall) : ansis.red(overall)
      console.log(`  ${modelLabel.padEnd(44)}${vpCells.join('')}  ${overallColor}`)
    }
  } else {
    // Fallback: no viewports detected
    console.log(`  ${ansis.bold('Model'.padEnd(44))} ${ansis.bold('Pass Rate'.padEnd(16))} ${ansis.bold('Score')}`)
    for (const run of results) {
      const pct = run.totalCount > 0 ? Math.round((run.passedCount / run.totalCount) * 100) : 0
      const status = run.passedCount === run.totalCount
        ? ansis.green(`${run.passedCount}/${run.totalCount} (${pct}%)`)
        : ansis.red(`${run.passedCount}/${run.totalCount} (${pct}%)`)
      const modelLabel = ansis.bold(`${run.provider}:${run.model}`)
      console.log(`  ${modelLabel.padEnd(44)} ${status.padEnd(16)} ${run.averageScore.toFixed(2)}`)
    }
  }
  console.log()
}

/**
 * Format results as JSON for machine-readable output
 */
export function formatJson(results: EvalRunResult[]): string {
  return JSON.stringify(results, null, 2)
}

// =============================================================================
// Helpers
// =============================================================================

/**
 * Abbreviate a message for display, collapsing XML tags and long content
 */
function abbreviateMessage(content: string, maxLen: number): string {
  // Collapse common XML wrapper tags to show just the inner content hint
  let abbreviated = content

  // Collapse <session_context>...</session_context>
  abbreviated = abbreviated.replace(/<session_context>[\s\S]*?<\/session_context>/g, '<session_context>…</session_context>')

  // Collapse <results>...</results>
  abbreviated = abbreviated.replace(/<results>[\s\S]*?<\/results>/g, '<results>…</results>')

  // Collapse <agent_mode>...</agent_mode>
  abbreviated = abbreviated.replace(/<agent_mode[^>]*>[\s\S]*?<\/agent_mode>/g, '<agent_mode>…</agent_mode>')

  // Collapse <user>...</user> to show just the inner text
  abbreviated = abbreviated.replace(/<user[^>]*>\s*([\s\S]*?)\s*<\/user>/g, '$1')

  // Trim and truncate
  abbreviated = abbreviated.trim()
  if (abbreviated.length > maxLen) {
    abbreviated = abbreviated.slice(0, maxLen) + '…'
  }

  return abbreviated
}
