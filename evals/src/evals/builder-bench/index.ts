/**
 * Builder Bench — benchmarks agent ability to iteratively fix bugs in real projects.
 *
 * Each scenario runs inside a Docker container with a broken project and test suite.
 * The agent uses tools to diagnose and fix bugs, then verifies tests pass.
 *
 * Dimensions:
 *   - Strategy: js-act, xml-act, native-openai
 *   - Toolset: fs-only, shell-only, fs-shell
 *   - Scenario: broken-sort, off-by-one-python, multi-file-import
 *
 * Scenario IDs: {strategy}:{toolset}/{scenario-id}
 * Variants: {strategy}:{toolset}
 */

import type { RunnableEval, EvalVariant, Scenario, ScenarioResult, ModelSpec, Check } from '../../types'
import { loadScenarios, type ScenarioDef } from './scenarios/index'
import { buildImage, createContainer, removeContainer, execInContainer, hashFiles, snapshotFiles, diffSnapshots, checkDocker } from './docker'
import { createDockerTools } from './tool-bridge'
import { runAgentLoop } from './agent-loop'
import type { StrategyId, ToolsetId } from './types'

// =============================================================================
// Constants
// =============================================================================

const STRATEGIES: StrategyId[] = ['js-act', 'xml-act', 'antml', 'native-openai']
const TOOLSETS: ToolsetId[] = ['fs-only', 'shell-only', 'fs-shell']
const MAX_TURNS = 15

// =============================================================================
// Scenario ID Parsing
// =============================================================================

function parseScenarioId(id: string): { strategy: StrategyId; toolset: ToolsetId; scenarioId: string } {
  // Format: {strategy}:{toolset}/{scenarioId}
  const slashIdx = id.indexOf('/')
  if (slashIdx === -1) throw new Error(`Invalid scenario ID: ${id}`)

  const prefix = id.slice(0, slashIdx)
  const scenarioId = id.slice(slashIdx + 1)

  const colonIdx = prefix.indexOf(':')
  if (colonIdx === -1) throw new Error(`Invalid scenario prefix: ${prefix}`)

  const strategy = prefix.slice(0, colonIdx) as StrategyId
  const toolset = prefix.slice(colonIdx + 1) as ToolsetId

  return { strategy, toolset, scenarioId }
}

// =============================================================================
// Scenario Building
// =============================================================================

function buildScenarios(scenarioDefs: ScenarioDef[]): Scenario[] {
  const scenarios: Scenario[] = []

  for (const strategy of STRATEGIES) {
    for (const toolset of TOOLSETS) {
      for (const def of scenarioDefs) {
        const checks: Check[] = [
          {
            id: 'tests-pass',
            description: 'All tests pass after agent fixes',
            evaluate: () => ({ passed: false, message: 'Not evaluated (runScenario handles this)' }),
          },
          {
            id: 'tests-intact',
            description: 'Test files were not modified by the agent',
            evaluate: () => ({ passed: false, message: 'Not evaluated (runScenario handles this)' }),
          },
        ]

        scenarios.push({
          id: `${strategy}:${toolset}/${def.id}`,
          description: `[${strategy} ${toolset}] ${def.description}`,
          messages: [{ role: 'user', content: [def.taskPrompt] }],
          checks,
        })
      }
    }
  }

  return scenarios
}

function buildVariants(scenarioDefs: ScenarioDef[]): EvalVariant[] {
  const variants: EvalVariant[] = []

  for (const strategy of STRATEGIES) {
    for (const toolset of TOOLSETS) {
      variants.push({
        id: `${strategy}:${toolset}`,
        label: `${strategy} ${toolset}`,
        count: scenarioDefs.length,
      })
    }
  }

  return variants
}

// =============================================================================
// Scenario Execution
// =============================================================================

function makeFail(scenarioId: string, message: string): ScenarioResult {
  return {
    scenarioId,
    checks: {
      'tests-pass': { passed: false, message },
      'tests-intact': { passed: false, message: 'Skipped due to earlier failure' },
    },
    passed: false,
    score: 0,
    rawResponse: JSON.stringify({ error: message }),
  }
}

async function executeScenario(
  scenario: Scenario,
  modelSpec: ModelSpec,
  scenarioDefs: ScenarioDef[],
): Promise<ScenarioResult> {
  // Pre-flight check
  if (!await checkDocker()) {
    return makeFail(scenario.id, 'Docker is not available. Install Docker and ensure the daemon is running.')
  }

  const { strategy, toolset, scenarioId } = parseScenarioId(scenario.id)

  const scenarioDef = scenarioDefs.find(d => d.id === scenarioId)
  if (!scenarioDef) return makeFail(scenario.id, `Unknown scenario: ${scenarioId}`)

  // 1. Build Docker image (cached)
  const imageTag = `builder-bench-${scenarioId}`
  try {
    await buildImage(scenarioDef.scenarioDir, imageTag)
  } catch (error) {
    return makeFail(scenario.id, `Docker build failed: ${error instanceof Error ? error.message : String(error)}`)
  }

  // 2. Create container
  const container = await createContainer(imageTag, '/workspace')

  try {
    // 3. Snapshot protected files + all source files
    const protectedHashes = await hashFiles(container, scenarioDef.protectedFiles)
    const beforeSnapshot = await snapshotFiles(container)

    // 4. Create Docker-bridged tools
    const toolBridge = createDockerTools(container, toolset)

    // 5. Run agent loop (system prompt is built by the real Cortex)
    const loopResult = await runAgentLoop({
      taskPrompt: scenarioDef.taskPrompt,
      strategy,
      modelSpec,
      toolBridge,
      maxTurns: MAX_TURNS,
      workDir: container.workDir,
      label: scenarioId,
    })

    // 7. Verify test integrity
    const postHashes = await hashFiles(container, scenarioDef.protectedFiles)
    const modifiedFiles: string[] = []
    for (const file of scenarioDef.protectedFiles) {
      if (protectedHashes[file] !== postHashes[file]) {
        modifiedFiles.push(file)
      }
    }
    const testsIntact = modifiedFiles.length === 0

    // 8. Capture diff of all changes the agent made
    const afterSnapshot = await snapshotFiles(container)
    const diff = diffSnapshots(beforeSnapshot, afterSnapshot)

    // 9. Run verify command
    const verifyResult = await execInContainer(container, scenarioDef.verifyCommand, 60_000)
    const testsPassed = verifyResult.exitCode === 0

    // 10. Build result
    const passed = testsPassed && testsIntact

    return {
      scenarioId: scenario.id,
      checks: {
        'tests-pass': {
          passed: testsPassed,
          message: testsPassed ? undefined : `Tests failed (exit ${verifyResult.exitCode})`,
          snippet: testsPassed ? undefined : (verifyResult.stderr || verifyResult.stdout).slice(0, 500),
        },
        'tests-intact': {
          passed: testsIntact,
          message: testsIntact ? undefined : `Protected files modified: ${modifiedFiles.join(', ')}`,
        },
      },
      passed,
      score: (Number(testsPassed) + Number(testsIntact)) / 2,
      rawResponse: JSON.stringify({
        turnCount: loopResult.turnCount,
        wallTimeMs: loopResult.wallTimeMs,
        usage: loopResult.usage,
        agentDone: loopResult.agentDone,
        testsIntact,
        modifiedProtectedFiles: modifiedFiles,
        testsPassed,
        diff,
        verifyStdout: verifyResult.stdout.slice(0, 2000),
        verifyStderr: verifyResult.stderr.slice(0, 2000),
      }),
    }
  } finally {
    // 10. Cleanup
    await removeContainer(container)
  }
}

// =============================================================================
// Eval Export
// =============================================================================

const scenarioDefs = loadScenarios()
const scenarios = buildScenarios(scenarioDefs)
const variants = buildVariants(scenarioDefs)

export const builderBenchEval: RunnableEval = {
  id: 'builder-bench',
  name: 'Builder Bench',
  description: `Benchmarks agent bug-fixing across ${STRATEGIES.length} strategies × ${TOOLSETS.length} toolsets × ${scenarioDefs.length} scenarios (${scenarios.length} total)`,
  scenarios,
  variants,
  defaultConcurrency: 2, // Docker containers are resource-heavy

  async runScenario(scenario: Scenario, modelSpec: ModelSpec): Promise<ScenarioResult> {
    return executeScenario(scenario, modelSpec, scenarioDefs)
  },
}
