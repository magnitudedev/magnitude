/**
 * Results persistence — save eval results to timestamped directories
 * as readable markdown files.
 */

import { mkdirSync, writeFileSync } from 'fs'
import { join } from 'path'
import type { Scenario, ScenarioResult, EvalRunResult } from './types'
import { chatMessageToDisplayText } from './message-content'

// =============================================================================
// Results directory management
// =============================================================================

/**
 * Create a timestamped results directory and return the path
 */
export function createResultsDir(baseDir: string = 'results'): string {
  const now = new Date()
  const timestamp = now.toISOString()
    .replace(/:/g, '-')
    .replace(/\.\d{3}Z$/, '')
  const dir = join(baseDir, timestamp)
  mkdirSync(dir, { recursive: true })
  return dir
}

// =============================================================================
// Markdown generation
// =============================================================================

/**
 * Generate a full markdown report for a single model's eval run
 */
export function generateModelReport(
  result: EvalRunResult,
  scenarios: Scenario[]
): string {
  const lines: string[] = []
  const pct = result.totalCount > 0 ? Math.round((result.passedCount / result.totalCount) * 100) : 0

  lines.push(`# Eval: ${result.evalId}`)
  lines.push(`## Model: ${result.provider}:${result.model}`)
  lines.push('')
  lines.push(`**Result: ${result.passedCount}/${result.totalCount} passed (${pct}%) · Score: ${result.averageScore.toFixed(2)}**`)
  lines.push('')

  // Per-variant summary (if scenarios have variant prefixes)
  const variants = new Map<string, ScenarioResult[]>()
  for (const sr of result.scenarios) {
    const slash = sr.scenarioId.indexOf('/')
    const variant = slash !== -1 ? sr.scenarioId.slice(0, slash) : ''
    if (!variants.has(variant)) variants.set(variant, [])
    variants.get(variant)!.push(sr)
  }

  if (variants.size > 1) {
    lines.push('## Variants')
    lines.push('')
    lines.push('| Variant | Passed | Total | Rate | Score |')
    lines.push('|---------|--------|-------|------|-------|')
    for (const [variant, srs] of variants) {
      const passed = srs.filter(s => s.passed).length
      const vpct = srs.length > 0 ? Math.round((passed / srs.length) * 100) : 0
      const avgScore = srs.length > 0 ? srs.reduce((sum, sr) => sum + sr.score, 0) / srs.length : 0
      lines.push(`| **${variant}** | ${passed} | ${srs.length} | ${vpct}% | ${avgScore.toFixed(2)} |`)
    }
    lines.push('')
  }

  lines.push('---')
  lines.push('')

  // Summary table
  lines.push('## Summary')
  lines.push('')
  lines.push('| Scenario | Result | Score | Failed Checks |')
  lines.push('|----------|--------|-------|---------------|')
  for (const sr of result.scenarios) {
    const icon = sr.passed ? '✓' : '✗'
    const failedChecks = Object.entries(sr.checks)
      .filter(([, c]) => !c.passed)
      .map(([id, c]) => `${id}: ${c.message ?? 'failed'}`)
    lines.push(`| ${sr.scenarioId} | ${icon} | ${sr.score.toFixed(2)} | ${failedChecks.join('; ') || '—'} |`)
  }
  lines.push('')
  lines.push('---')
  lines.push('')

  // Detailed results per scenario
  lines.push('## Detailed Results')
  lines.push('')

  for (const sr of result.scenarios) {
    const scenario = scenarios.find(s => s.id === sr.scenarioId)
    lines.push(`### ${sr.scenarioId}`)
    lines.push('')
    lines.push(`**Score:** ${sr.score.toFixed(2)}`)
    lines.push('')
    if (scenario) {
      lines.push(`> ${scenario.description}`)
      lines.push('')
    }

    // Conversation
    if (scenario) {
      lines.push('#### Conversation')
      lines.push('')
      for (const msg of scenario.messages) {
        const role = msg.role.toUpperCase()
        const content = abbreviateForMarkdown(chatMessageToDisplayText(msg))
        lines.push(`**${role}:**`)
        lines.push('')
        if (content.includes('\n')) {
          lines.push('```')
          lines.push(content)
          lines.push('```')
        } else {
          lines.push(content)
        }
        lines.push('')
      }
    }

    // Response
    lines.push('#### LLM Response')
    lines.push('')
    if (sr.rawResponse) {
      lines.push('```javascript')
      lines.push(sr.rawResponse)
      lines.push('```')
    } else {
      lines.push('*(empty response)*')
    }
    lines.push('')

    // Check results
    lines.push('#### Checks')
    lines.push('')
    for (const [checkId, check] of Object.entries(sr.checks)) {
      const icon = check.passed ? '✓' : '✗'
      const checkScore = check.score ?? (check.passed ? 1 : 0)
      lines.push(`- ${icon} **${checkId}** (score: ${checkScore.toFixed(2)})${check.passed ? '' : ': ' + (check.message ?? 'failed')}`)
      if (check.snippet) {
        lines.push('  ```')
        lines.push('  ' + check.snippet.split('\n').join('\n  '))
        lines.push('  ```')
      }
    }
    lines.push('')
    lines.push('---')
    lines.push('')
  }

  return lines.join('\n')
}

/**
 * Generate a combined summary markdown for multiple model runs
 */
export function generateSummaryReport(results: EvalRunResult[]): string {
  const lines: string[] = []

  lines.push('# Eval Results Summary')
  lines.push('')
  lines.push(`*Generated: ${new Date().toISOString()}*`)
  lines.push('')

  // Overview table
  lines.push('## Overview')
  lines.push('')
  lines.push('| Model | Passed | Total | Rate | Score |')
  lines.push('|-------|--------|-------|------|-------|')
  for (const r of results) {
    const pct = r.totalCount > 0 ? Math.round((r.passedCount / r.totalCount) * 100) : 0
    lines.push(`| ${r.provider}:${r.model} | ${r.passedCount} | ${r.totalCount} | ${pct}% | ${r.averageScore.toFixed(2)} |`)
  }
  lines.push('')

  // Per-variant breakdown (if scenarios have variant prefixes)
  if (results.length > 0) {
    const variants = new Map<string, string[]>()
    for (const sr of results[0].scenarios) {
      const slash = sr.scenarioId.indexOf('/')
      const variant = slash !== -1 ? sr.scenarioId.slice(0, slash) : ''
      if (!variants.has(variant)) variants.set(variant, [])
      variants.get(variant)!.push(sr.scenarioId)
    }

    if (variants.size > 1) {
      lines.push('## Per-Variant Breakdown')
      lines.push('')
      const header = ['Variant', ...results.map(r => `${r.provider}:${r.model}`)]
      lines.push('| ' + header.join(' | ') + ' |')
      lines.push('| ' + header.map(() => '---').join(' | ') + ' |')
      for (const [variant, ids] of variants) {
        const cells = results.map(r => {
          const scores = ids
            .map(id => r.scenarios.find(s => s.scenarioId === id)?.score)
            .filter((score): score is number => score != null)
          const avgScore = scores.length > 0 ? scores.reduce((sum, score) => sum + score, 0) / scores.length : 0
          return avgScore.toFixed(2)
        })
        lines.push(`| **${variant}** | ${cells.join(' | ')} |`)
      }
      lines.push('')
    }
  }

  // Per-scenario breakdown
  if (results.length > 0) {
    const scenarioIds = results[0].scenarios.map(s => s.scenarioId)
    lines.push('## Per-Scenario Breakdown')
    lines.push('')
    const header = ['Scenario', ...results.map(r => `${r.provider}:${r.model}`)]
    lines.push('| ' + header.join(' | ') + ' |')
    lines.push('| ' + header.map(() => '---').join(' | ') + ' |')
    for (const sid of scenarioIds) {
      const cells = results.map(r => {
        const sr = r.scenarios.find(s => s.scenarioId === sid)
        return sr ? sr.score.toFixed(2) : '—'
      })
      lines.push(`| ${sid} | ${cells.join(' | ')} |`)
    }
    lines.push('')
  }

  // Ranking by overall score
  if (results.length > 1) {
    const ranked = [...results].sort((a, b) => b.averageScore - a.averageScore)

    lines.push('## Ranking (by Overall Score)')
    lines.push('')
    lines.push('| Rank | Model | Distance Score | Hit Rate | Overall |')
    lines.push('|------|-------|---------------|----------|---------|')

    for (let i = 0; i < ranked.length; i++) {
      const r = ranked[i]
      const model = `${r.provider}:${r.model}`

      // Separate distance checks vs hit checks
      let distanceTotal = 0
      let distanceCount = 0
      let hitPassed = 0
      let hitCount = 0

      for (const sr of r.scenarios) {
        for (const [checkId, check] of Object.entries(sr.checks)) {
          if (checkId.endsWith('-distance')) {
            distanceTotal += check.score ?? (check.passed ? 1 : 0)
            distanceCount++
          } else if (checkId.endsWith('-hit')) {
            hitCount++
            if (check.passed) hitPassed++
          }
        }
      }

      const distScore = distanceCount > 0 ? (distanceTotal / distanceCount).toFixed(3) : '—'
      const hitRate = hitCount > 0 ? `${hitPassed}/${hitCount} (${Math.round((hitPassed / hitCount) * 100)}%)` : '—'

      lines.push(`| ${i + 1} | ${model} | ${distScore} | ${hitRate} | ${r.averageScore.toFixed(3)} |`)
    }
    lines.push('')
  }

  // Raw check details per model
  if (results.length > 0) {
    lines.push('## Raw Results')
    lines.push('')

    for (const r of results) {
      const model = `${r.provider}:${r.model}`
      lines.push(`### ${model}`)
      lines.push('')
      lines.push(`Overall Score: ${r.averageScore.toFixed(3)} | Passed: ${r.passedCount}/${r.totalCount}`)
      lines.push('')

      for (const sr of r.scenarios) {
        lines.push(`**${sr.scenarioId}** (score: ${sr.score.toFixed(3)})`)
        lines.push('')
        for (const [checkId, check] of Object.entries(sr.checks)) {
          const icon = check.passed ? '✓' : '✗'
          const score = check.score ?? (check.passed ? 1 : 0)
          lines.push(`- ${icon} ${checkId} (${score.toFixed(3)}) ${check.message ?? ''}`)
        }
        lines.push('')
      }

      lines.push('---')
      lines.push('')
    }
  }

  return lines.join('\n')
}

/**
 * Save all results to a directory.
 * If `dir` is provided, writes there. Otherwise creates a new timestamped dir.
 */
export function saveResults(
  results: EvalRunResult[],
  scenarios: Scenario[],
  dir?: string,
): string {
  const resultsDir = dir ?? createResultsDir()

  // Save per-model reports
  for (const result of results) {
    const filename = `${result.provider}-${result.model}`.replace(/[/:]/g, '-') + '.md'
    const report = generateModelReport(result, scenarios)
    writeFileSync(join(resultsDir, filename), report, 'utf-8')
  }

  // Save combined summary
  if (results.length > 0) {
    const summary = generateSummaryReport(results)
    writeFileSync(join(resultsDir, 'summary.md'), summary, 'utf-8')
  }

  // Save raw JSON
  writeFileSync(join(resultsDir, 'results.json'), JSON.stringify(results, null, 2), 'utf-8')

  return resultsDir
}

// =============================================================================
// Helpers
// =============================================================================

function abbreviateForMarkdown(content: string): string {
  let abbreviated = content

  // Collapse XML tags
  abbreviated = abbreviated.replace(/<session_context>[\s\S]*?<\/session_context>/g, '[session context]')
  abbreviated = abbreviated.replace(/<results>[\s\S]*?<\/results>/g, '[tool results]')
  abbreviated = abbreviated.replace(/<agent_mode[^>]*>[\s\S]*?<\/agent_mode>/g, '[agent mode]')
  abbreviated = abbreviated.replace(/<reminder>[\s\S]*?<\/reminder>/g, '')

  // Extract user message content
  abbreviated = abbreviated.replace(/<user[^>]*>\s*([\s\S]*?)\s*<\/user>/g, '$1')

  return abbreviated.trim()
}
