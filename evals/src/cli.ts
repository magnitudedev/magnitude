#!/usr/bin/env bun
/**
 * Evals CLI — run LLM evaluations interactively or via command line
 */

import { Command } from '@commander-js/extra-typings'
import * as clack from '@clack/prompts'
import { detectProviders } from '@magnitudedev/providers'
import ansis from 'ansis'
import { readdirSync, existsSync, readFileSync } from 'fs'
import { join } from 'path'
// import { proseEval } from './evals/prose/index' // TODO: broken — js-act removed
// import { reactEditEval } from './evals/react-edit/index' // TODO: broken — js-act removed
// import { builderBenchEval } from './evals/builder-bench/index' // TODO: broken — strategy system removed
// import { formatCompareEval } from './evals/format-compare/index' // TODO: broken — js-act removed
import { visualGroundingEval } from './evals/visual-grounding/index'
import { xpathEval } from './evals/xpath/index'
import { geminiGroundingEval } from './evals/gemini-grounding/index'
import { oneShotsEval } from './evals/one-shots/index'
import { leadDispatchEval } from './evals/lead-dispatch/index'
import { behaviorEval } from './evals/behavior/index'
import { memoryExtractionEval } from './evals/memory-extraction/index'
import { xmlNewlinesEval } from './evals/xml-newlines/index'
import { entityEscapingEval } from './evals/entity-escaping/index'
import { runEval, type RunCallbacks } from './runner'
import {
  printModelHeader,
  printScenarioResult,
  printModelSummary,
  printReport,
  formatJson
} from './report'
import { createResultsDir, saveResults } from './results'
import { parseModelSpec } from './types'
import type { Eval, ModelSpec, EvalRunResult, ScenarioResult, Scenario } from './types'

function loadIgnoredModels(): Set<string> {
  const configPath = join(import.meta.dir, '../eval-config.json')
  try {
    const raw = JSON.parse(readFileSync(configPath, 'utf-8'))
    return new Set(raw.ignoredModels ?? [])
  } catch {
    return new Set()
  }
}

// =============================================================================
// Available evals registry
// =============================================================================

const EVALS: Record<string, Eval> = {
  // prose: proseEval, // TODO: broken — js-act removed
  // 'react-edit': reactEditEval, // TODO: broken — js-act removed
  // 'builder-bench': builderBenchEval, // TODO: broken — strategy system removed
  // 'format-compare': formatCompareEval, // TODO: broken — js-act removed
  'visual-grounding': visualGroundingEval,
  xpath: xpathEval,
  'gemini-grounding': geminiGroundingEval,
  'one-shots': oneShotsEval,
  'lead-dispatch': leadDispatchEval,
  behavior: behaviorEval,
  'memory-extraction': memoryExtractionEval,
  'xml-newlines': xmlNewlinesEval,
  'entity-escaping': entityEscapingEval,
}

// =============================================================================
// Default models to offer in interactive mode
// =============================================================================

const DEFAULT_MODELS: { label: string; value: string; hint?: string }[] = [
  { label: 'Claude Sonnet 4.6 (Anthropic)', value: 'anthropic:claude-sonnet-4-6' },
  { label: 'Claude Haiku 4.5 (Anthropic)', value: 'anthropic:claude-haiku-4-5' },
  { label: 'GPT-5.3 Codex (OpenAI)', value: 'openai:gpt-5.3-codex' },
  { label: 'GPT-5.3 Codex Spark (OpenAI)', value: 'openai:gpt-5.3-codex-spark' },
  { label: 'Gemini 3.1 Pro Preview (Google)', value: 'google:gemini-3.1-pro-preview' },
  { label: 'Gemini 3 Flash Preview (Google)', value: 'google:gemini-3-flash-preview' },
  { label: 'MiniMax M2.5', value: 'minimax:MiniMax-M2.5' },
  { label: 'GLM-4.7 (Z.AI)', value: 'zai:glm-4.7' },
  { label: 'Gemma 3n E4B (Local)', value: 'local:google/gemma-3n-e4b' },
]

function getAvailableModels() {
  const detected = detectProviders()
  const connectedProviderIds = new Set(detected.map(d => d.provider.id))
  return DEFAULT_MODELS.map(m => {
    const providerId = m.value.split(':')[0]
    const connected = connectedProviderIds.has(providerId)
    return {
      ...m,
      hint: connected ? '✓ connected' : '✗ not connected',
      connected,
    }
  })
}

// =============================================================================
// CLI Definition
// =============================================================================

const program = new Command()
  .name('eval')
  .description('Run LLM evaluations')
  .version('0.1.0')

// List available evals
program
  .command('list')
  .description('List available evaluations')
  .action(() => {
    console.log(ansis.bold.underline('Available Evaluations'))
    console.log()
    for (const [id, eval_] of Object.entries(EVALS)) {
      console.log(`  ${ansis.bold(id)} — ${eval_.description}`)
      console.log(`    ${eval_.scenarios.length} scenarios`)
      console.log()
    }
  })
// Show eval results
program
  .command('results')
  .description('Show latest eval results across runs')
  .argument('[eval-id]', 'Eval to show results for')
  .action(async (evalId) => {
    // Resolve eval
    let selectedEvalId: string

    if (evalId) {
      if (!EVALS[evalId]) {
        console.error(ansis.red(`Unknown eval: ${evalId}`))
        console.error(`Available: ${Object.keys(EVALS).join(', ')}`)
        process.exit(1)
      }
      selectedEvalId = evalId
    } else {
      clack.intro(ansis.bold('Eval Results Viewer'))

      const evalChoice = await clack.select({
        message: 'Which eval would you like to view results for?',
        options: Object.entries(EVALS).map(([id, e]) => ({
          label: `${e.name} — ${e.description}`,
          value: id
        }))
      })

      if (clack.isCancel(evalChoice)) {
        clack.cancel('Cancelled')
        process.exit(0)
      }

      selectedEvalId = evalChoice as string
    }

    const selectedEval = EVALS[selectedEvalId]

    // Scan all result directories
    const baseDir = 'results'
    if (!existsSync(baseDir)) {
      console.log(ansis.yellow('No results directory found.'))
      process.exit(0)
    }

    const dirs = readdirSync(baseDir, { withFileTypes: true })
      .filter(d => d.isDirectory())
      .map(d => d.name)
      .sort() // chronological since names are ISO timestamps

    // Build expected variant scenario counts
    const variantExpectedCounts = new Map<string, number>()
    if (selectedEval.variants && selectedEval.variants.length > 0) {
      for (const v of selectedEval.variants) {
        variantExpectedCounts.set(v.id, v.count)
      }
    }

    // Collect latest complete result per model×variant
    // Key: `provider:model::variantId`
    const latestByModelVariant = new Map<string, { result: EvalRunResult; scenarios: ScenarioResult[]; dir: string }>()

    for (const dir of dirs) {
      const dirPath = join(baseDir, dir)

      // Collect all EvalRunResult objects from this directory
      // Check results.json first, then fall back to individual per-model JSON files
      const runResults: EvalRunResult[] = []

      const resultsJsonPath = join(dirPath, 'results.json')
      if (existsSync(resultsJsonPath)) {
        try {
          const raw = JSON.parse(require('fs').readFileSync(resultsJsonPath, 'utf-8'))
          if (Array.isArray(raw)) {
            runResults.push(...raw)
          }
        } catch { /* skip malformed */ }
      }

      // Also scan for individual per-model JSON files (written incrementally by runner)
      try {
        const files = readdirSync(dirPath).filter(f => f.endsWith('.json') && f !== 'results.json')
        for (const file of files) {
          try {
            const raw = JSON.parse(require('fs').readFileSync(join(dirPath, file), 'utf-8')) as EvalRunResult
            // Avoid duplicates — skip if results.json already has this model
            if (runResults.some(r => r.provider === raw.provider && r.model === raw.model && r.evalId === raw.evalId)) continue
            runResults.push(raw)
          } catch { /* skip malformed */ }
        }
      } catch { /* skip unreadable dirs */ }

      for (const result of runResults) {
        if (result.evalId !== selectedEvalId) continue
        const modelKey = `${result.provider}:${result.model}`

        // Group scenarios by variant
        const byVariant = new Map<string, ScenarioResult[]>()
        for (const sr of result.scenarios) {
          const slash = sr.scenarioId.indexOf('/')
          const variant = slash !== -1 ? sr.scenarioId.slice(0, slash) : ''
          if (!byVariant.has(variant)) byVariant.set(variant, [])
          byVariant.get(variant)!.push(sr)
        }

        for (const [variant, scenarios] of byVariant) {
          const expected = variantExpectedCounts.get(variant) ?? selectedEval.scenarios.length
          if (scenarios.length < expected) continue // incomplete variant run
          const key = `${modelKey}::${variant}`
          latestByModelVariant.set(key, { result, scenarios, dir })
        }
      }
    }

    if (latestByModelVariant.size === 0) {
      console.log(ansis.yellow(`No complete results found for eval "${selectedEvalId}".`))
      process.exit(0)
    }

    // Aggregate per model, filtering out ignored models from eval-config.json
    const ignoredModels = loadIgnoredModels()
    const modelKeys = [...new Set([...latestByModelVariant.keys()].map(k => k.split('::')[0]))]
      .filter(k => !ignoredModels.has(k))
      .sort()

    // Overview table
    console.log()
    console.log(ansis.bold.underline(`Results: ${selectedEval.name}`))
    console.log()

    console.log(ansis.bold('Overview'))
    console.log()
    console.log('  Model                                          Passed   Total   Rate    Score')
    console.log('  ' + '─'.repeat(84))
    for (const modelKey of modelKeys) {
      let totalPassed = 0
      let totalCount = 0
      let totalScore = 0
      for (const [key, { scenarios }] of latestByModelVariant) {
        if (key.split('::')[0] !== modelKey) continue
        totalPassed += scenarios.filter(s => s.passed).length
        totalCount += scenarios.length
        totalScore += scenarios.reduce((sum, s) => sum + s.score, 0)
      }
      const pct = totalCount > 0 ? Math.round((totalPassed / totalCount) * 100) : 0
      const avgScore = totalCount > 0 ? totalScore / totalCount : 0
      const pctStr = pct === 100 ? ansis.green(`${pct}%`) : pct === 0 ? ansis.red(`${pct}%`) : ansis.yellow(`${pct}%`)
      const model = modelKey.padEnd(48)
      const passed = String(totalPassed).padStart(4)
      const total = String(totalCount).padStart(5)
      const score = avgScore.toFixed(2).padStart(7)
      console.log(`  ${model} ${passed}   ${total}   ${pctStr}  ${score}`)
    }
    console.log()
    if (selectedEval.variants && selectedEval.variants.length > 0) {
      const variantIds = selectedEval.variants.map(v => v.id)

      console.log(ansis.bold('Per-Variant Overview'))
      console.log()
      console.log('  Variant                                        Passed   Total   Rate    Score')
      console.log('  ' + '─'.repeat(84))
      for (const variantId of variantIds) {
        let totalPassed = 0
        let totalCount = 0
        let totalScore = 0

        for (const modelKey of modelKeys) {
          const entry = latestByModelVariant.get(`${modelKey}::${variantId}`)
          if (!entry) continue
          totalPassed += entry.scenarios.filter(s => s.passed).length
          totalCount += entry.scenarios.length
          totalScore += entry.scenarios.reduce((sum, s) => sum + s.score, 0)
        }

        const pct = totalCount > 0 ? Math.round((totalPassed / totalCount) * 100) : 0
        const avgScore = totalCount > 0 ? totalScore / totalCount : 0
        const pctStr = pct === 100 ? ansis.green(`${pct}%`) : pct === 0 ? ansis.red(`${pct}%`) : ansis.yellow(`${pct}%`)
        const variant = variantId.padEnd(46)
        const passed = String(totalPassed).padStart(4)
        const total = String(totalCount).padStart(5)
        const score = avgScore.toFixed(2).padStart(7)
        console.log(`  ${variant} ${passed}   ${total}   ${pctStr}  ${score}`)
      }
      console.log()
    }


    // Per-variant breakdown (if applicable)
    if (selectedEval.variants && selectedEval.variants.length > 0) {
      const variantIds = selectedEval.variants.map(v => v.id)

      // Model column headers (short names)
      const modelHeaders = modelKeys.map(k => {
        const short = k.length > 20 ? k.slice(k.indexOf(':') + 1) : k
        return short.length > 20 ? short.slice(0, 20) : short
      })

      // Split variant IDs into dimensions using `:` separator.
      // If variants have multiple dimensions (e.g. "xml-hybrid:r0"),
      // group by the last dimension and use the prefix as the row label.
      // If variants have no `:`, render a single flat table.
      const hasDimensions = variantIds.some(v => v.includes(':'))

      if (hasDimensions) {
        // Group variants: last segment after `:` → table key, prefix → row label
        const tables = new Map<string, { rowLabel: string; variantId: string }[]>()
        for (const variantId of variantIds) {
          const lastColon = variantId.lastIndexOf(':')
          const tableKey = variantId.slice(lastColon + 1)
          const rowLabel = variantId.slice(0, lastColon)
          if (!tables.has(tableKey)) tables.set(tableKey, [])
          tables.get(tableKey)!.push({ rowLabel, variantId })
        }

        for (const [tableKey, rows] of tables) {
          console.log(ansis.bold(tableKey))
          console.log()
          console.log('  ' + 'Format'.padEnd(30) + modelHeaders.map(h => h.padStart(22)).join(''))
          console.log('  ' + '─'.repeat(30 + modelHeaders.length * 22))

          for (const { rowLabel, variantId } of rows) {
            const cells = modelKeys.map(modelKey => {
              const key = `${modelKey}::${variantId}`
              const entry = latestByModelVariant.get(key)
              if (!entry) return '—'.padStart(22)
              const passed = entry.scenarios.filter(s => s.passed).length
              const pct = Math.round((passed / entry.scenarios.length) * 100)
              const avgScore = entry.scenarios.reduce((sum, s) => sum + s.score, 0) / entry.scenarios.length
              const str = `${passed}/${entry.scenarios.length} (${pct}%) ${avgScore.toFixed(2)}`
              return str.padStart(22)
            })
            console.log('  ' + rowLabel.padEnd(30) + cells.join(''))
          }
          console.log()
        }
      } else {
        // Flat table — no dimension splitting
        console.log(ansis.bold('Per-Variant'))
        console.log()
        console.log('  ' + 'Variant'.padEnd(30) + modelHeaders.map(h => h.padStart(22)).join(''))
        console.log('  ' + '─'.repeat(30 + modelHeaders.length * 22))

        for (const variantId of variantIds) {
          const cells = modelKeys.map(modelKey => {
            const key = `${modelKey}::${variantId}`
            const entry = latestByModelVariant.get(key)
            if (!entry) return '—'.padStart(22)
            const passed = entry.scenarios.filter(s => s.passed).length
            const pct = Math.round((passed / entry.scenarios.length) * 100)
            const avgScore = entry.scenarios.reduce((sum, s) => sum + s.score, 0) / entry.scenarios.length
            const str = `${passed}/${entry.scenarios.length} (${pct}%) ${avgScore.toFixed(2)}`
            return str.padStart(22)
          })
          console.log('  ' + variantId.padEnd(30) + cells.join(''))
        }
        console.log()
      }
    }


    if (!evalId) {
      clack.outro('Done!')
    }
  })


// Run an eval
program
  .command('run')
  .description('Run an evaluation')
  .argument('[eval-id]', 'Eval to run (e.g., "prose")')
  .option('-m, --model <spec...>', 'Model(s) to test as provider:model')
  .option('-s, --scenario <id...>', 'Specific scenario(s) to run')
  .option('-v, --variant <id...>', 'Variant(s) to run (for evals with variants)')
  .option('--json', 'Output results as JSON')
  .option('--all-models', 'Run all default models')
  .option('-q, --quiet', 'Only show summary (no response details)')
  .option('-c, --concurrency <n>', 'Max parallel scenarios per model', '4')
  .option('--no-save', 'Do not save results to disk')
  .option('-r, --repeat <n>', 'Run each scenario N times per model', '1')
  .action(async (evalId, options) => {
    // Resolve which eval to run
    let selectedEval: Eval

    if (evalId) {
      if (!EVALS[evalId]) {
        console.error(ansis.red(`Unknown eval: ${evalId}`))
        console.error(`Available: ${Object.keys(EVALS).join(', ')}`)
        process.exit(1)
      }
      selectedEval = EVALS[evalId]
    } else {
      // Interactive eval selection
      clack.intro(ansis.bold('LLM Evaluation Runner'))

      const evalChoice = await clack.select({
        message: 'Which eval would you like to run?',
        options: Object.entries(EVALS).map(([id, e]) => ({
          label: `${e.name} — ${e.description}`,
          value: id
        }))
      })

      if (clack.isCancel(evalChoice)) {
        clack.cancel('Cancelled')
        process.exit(0)
      }

      selectedEval = EVALS[evalChoice as string]
    }

    // Resolve variants (filters scenarios by prefix)
    let scenarioFilter: string[] | undefined = options.scenario

    if (!scenarioFilter && selectedEval.variants && selectedEval.variants.length > 1) {
      if (options.variant && options.variant.length > 0) {
        // CLI flag — validate variant IDs
        const validIds = new Set(selectedEval.variants.map(v => v.id))
        for (const v of options.variant) {
          if (!validIds.has(v)) {
            console.error(ansis.red(`Unknown variant: ${v}`))
            console.error(`Available: ${[...validIds].join(', ')}`)
            process.exit(1)
          }
        }
        const selected = new Set(options.variant)
        scenarioFilter = selectedEval.scenarios
          .filter(s => selected.has(s.id.slice(0, s.id.indexOf('/'))))
          .map(s => s.id)
      } else if (options.model || options.allModels || selectedEval.defaultModels) {
        // Non-interactive: run all variants
        scenarioFilter = undefined
      } else {
        // Interactive variant selection
        const variantChoices = await clack.multiselect({
          message: 'Which variants would you like to run?',
          options: selectedEval.variants.map(v => ({
            label: v.label,
            value: v.id,
            hint: `${v.count} scenarios`,
          })),
          required: true,
        })

        if (clack.isCancel(variantChoices)) {
          clack.cancel('Cancelled')
          process.exit(0)
        }

        const selected = new Set(variantChoices as string[])
        scenarioFilter = selectedEval.scenarios
          .filter(s => selected.has(s.id.slice(0, s.id.indexOf('/'))))
          .map(s => s.id)
      }
    }

    // Resolve which models to test
    let modelSpecs: ModelSpec[]

    if (options.model && options.model.length > 0) {
      // Explicit -m flags always take priority
      modelSpecs = options.model.map(parseModelSpec)
    } else if (options.allModels || selectedEval.defaultModels) {
      // --all-models flag, or eval has a hardcoded model list (use it automatically)
      if (selectedEval.defaultModels && selectedEval.defaultModels.length > 0) {
        modelSpecs = selectedEval.defaultModels
        console.log(ansis.dim(`Using ${modelSpecs.length} models from ${selectedEval.id} default list`))
      } else {
        const available = getAvailableModels()
        const connected = available.filter(m => m.connected)
        if (connected.length === 0) {
          console.error(ansis.red('No providers connected. Run magnitude setup or set API keys.'))
          process.exit(1)
        }
        modelSpecs = connected.map(m => parseModelSpec(m.value))
      }
    } else {
      // Interactive model selection
      const available = getAvailableModels()
      const modelChoices = await clack.multiselect({
        message: 'Which models would you like to test?',
        options: available.map(m => ({
          label: m.label,
          value: m.value,
          hint: m.hint
        })),
        required: true
      })

      if (clack.isCancel(modelChoices)) {
        clack.cancel('Cancelled')
        process.exit(0)
      }

      modelSpecs = (modelChoices as string[]).map(parseModelSpec)
    }

    // Run the eval with progressive output
    const allResults: EvalRunResult[] = []
    const isQuiet = options.quiet || false
    const isJson = options.json || false
    const concurrency = parseInt(options.concurrency, 10)
    const repeat = parseInt(options.repeat, 10)
    const shouldSave = options.save !== false && !isJson

    // Create results dir upfront so incremental writes work
    const resultsDir = shouldSave ? createResultsDir() : null
    if (resultsDir) {
      console.log(ansis.dim(`Results → ${resultsDir}/`))
    }

    // Track completed count for progress
    let completedCount = 0

    for (const modelSpec of modelSpecs) {
      completedCount = 0

      const callbacks: RunCallbacks = isJson ? {} : {
        onModelStart(spec) {
          printModelHeader(spec.label)
        },
        onScenarioComplete(scenario, result, index, total) {
          completedCount++
          if (isQuiet) {
            const icon = result.passed ? ansis.green('✓') : ansis.red('✗')
            const failedChecks = Object.entries(result.checks)
              .filter(([, c]) => !c.passed)
              .map(([id, c]) => `${id}: ${c.message ?? 'failed'}`)

            const progress = ansis.dim(`[${completedCount}/${total}]`)
            process.stdout.write(`  ${progress} ${icon} ${scenario.id}`)
            if (failedChecks.length > 0) {
              const [firstLine, ...rest] = failedChecks[0].split('\n')
              process.stdout.write(ansis.dim(` — ${firstLine}`))
              console.log()
              for (const line of rest.slice(0, 5)) {
                console.log(ansis.dim(`         ${line}`))
              }
            } else {
              console.log()
            }
          } else {
            const progress = ansis.dim(`[${completedCount}/${total}]`)
            console.log(`${progress}`)
            printScenarioResult(scenario, result, index, total)
          }
        },
        onModelComplete(spec, result) {
          printModelSummary(result)
        }
      }

      try {
        const result = await runEval(selectedEval, modelSpec, {
          scenarios: scenarioFilter,
          callbacks,
          concurrency,
          repeat,
          resultsDir: resultsDir ?? undefined,
        })
        allResults.push(result)
      } catch (error) {
        console.error(ansis.red(`Error running ${modelSpec.label}: ${error instanceof Error ? error.message : String(error)}`))
      }
    }

    // Save final markdown reports
    if (resultsDir) {
      const scenarios = scenarioFilter
        ? selectedEval.scenarios.filter(s => scenarioFilter!.includes(s.id))
        : selectedEval.scenarios

      saveResults(allResults, scenarios, resultsDir)
      console.log(ansis.dim(`Results saved to ${resultsDir}/`))
    }

    // Final output
    if (isJson) {
      console.log(formatJson(allResults))
    } else if (allResults.length > 1) {
      printReport(allResults)
    }

    if (!evalId) {
      clack.outro('Done!')
    }
  })

// Default command — run interactively
program
  .action(async () => {
    await program.parseAsync(['node', 'eval', 'run'])
  })

program.parse()
