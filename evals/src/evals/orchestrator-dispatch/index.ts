/**
 * Orchestrator Dispatch Eval
 *
 * Tests orchestrator subagent dispatching across multiple turns.
 * Uses the real orchestrator system prompt, calls the model in a loop,
 * and injects synthetic agent responses to drive the orchestrator through
 * its full decision-making lifecycle.
 *
 * Agents create their own artifacts internally — the orchestrator doesn't
 * pre-create them. The turn loop attaches synthetic artifact content to
 * agent responses regardless of writable artifact IDs.
 *
 * When the orchestrator uses tools directly (fs-read, fs-search, shell, etc.),
 * the turn loop returns mock results from the scenario's mockFiles map.
 */

import type { RunnableEval, Scenario, ScenarioResult, ModelSpec, CheckResult, EvalVariant } from '../../types'
import type { ChatMessage } from '@magnitudedev/llm-core'
import { callModel } from '../../runner'
import { getAgentDefinition, generateXmlActToolDocs } from '@magnitudedev/agent'
import { getXmlActProtocol, builtInThinkingLenses } from '@magnitudedev/agent-definition'
import { parseOrchestratorResponse } from './xml-parser'
import type { ParsedOrchestratorResponse } from './xml-parser'
import { ALL_SCENARIOS, type DispatchScenario } from './scenarios'

// =============================================================================
// System prompt
// =============================================================================

let cachedSystemPrompt: string | null = null

function getSystemPrompt(): string {
  if (!cachedSystemPrompt) {
    const agentDef = getAgentDefinition('orchestrator')
    const toolDocs = generateXmlActToolDocs(agentDef, ['think'])
    cachedSystemPrompt = `${getXmlActProtocol('user', builtInThinkingLenses.slice())}\n\n${agentDef.systemPrompt}\n\n## Tools\n\n${toolDocs}`
  }
  return cachedSystemPrompt
}

// =============================================================================
// Synthetic response formatting
// =============================================================================

/** Counter for generating unique agent IDs within a scenario run */
let agentIdCounter = 0

function nextAgentId(type: string): string {
  return `${type}-${++agentIdCounter}`
}

/**
 * Build inspect refs and result lines for all direct tool uses.
 * Shared between buildTurnFeedback and buildDirectToolFeedback.
 */
function buildToolResults(
  parsed: ParsedOrchestratorResponse,
  mockFiles: Record<string, string>,
  artifactStore: Map<string, string>,
): { resultLines: string[]; inspectLines: string[] } {
  const resultLines: string[] = []
  const inspectLines: string[] = []

  // --- fs-read ---
  for (const fsRead of parsed.fsReads) {
    const content = mockFiles[fsRead.path]
    if (content !== undefined) {
      inspectLines.push(`<ref tool="${fsRead.refName}">${content}</ref>`)
    } else {
      resultLines.push(`<tool name="fs-read"><error>ENOENT: no such file or directory '${fsRead.path}'</error></tool>`)
    }
  }

  // --- fs-search ---
  for (const fsSearch of parsed.fsSearches) {
    const items: string[] = []
    let pattern: RegExp
    try {
      pattern = new RegExp(fsSearch.pattern, 'gi')
    } catch {
      pattern = new RegExp(fsSearch.pattern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi')
    }
    for (const [filePath, content] of Object.entries(mockFiles)) {
      if (fsSearch.path !== '.' && !filePath.startsWith(fsSearch.path.replace(/^\.\//, ''))) continue
      const lines = content.split('\n')
      for (let i = 0; i < lines.length; i++) {
        if (pattern.test(lines[i])) {
          items.push(`<item file="${filePath}">${i + 1}|${lines[i]}</item>`)
        }
        pattern.lastIndex = 0
      }
    }
    if (items.length > 0) {
      inspectLines.push(`<ref tool="${fsSearch.refName}">\n${items.join('\n')}\n</ref>`)
    } else {
      inspectLines.push(`<ref tool="${fsSearch.refName}">No matches found</ref>`)
    }
  }

  // --- fs-tree ---
  for (const fsTree of parsed.fsTrees) {
    const dirs = new Set<string>()
    const entries: string[] = []
    const prefix = fsTree.path === '.' ? '' : fsTree.path.replace(/^\.\//, '')
    for (const filePath of Object.keys(mockFiles).sort()) {
      if (prefix && !filePath.startsWith(prefix)) continue
      // Collect parent directories
      const parts = filePath.split('/')
      for (let i = 1; i < parts.length; i++) {
        const dir = parts.slice(0, i).join('/')
        if (!dirs.has(dir)) {
          dirs.add(dir)
          entries.push(`<entry path="${dir}" name="${parts[i - 1]}" type="dir" />`)
        }
      }
      entries.push(`<entry path="${filePath}" name="${parts[parts.length - 1]}" type="file" />`)
    }
    inspectLines.push(`<ref tool="${fsTree.refName}">\n${entries.join('\n')}\n</ref>`)
  }

  // --- shell ---
  for (const shell of parsed.shells) {
    const cmd = shell.command
    // Try to handle cat commands by looking up mock files
    const catMatch = cmd.match(/cat\s+(\S+)/)
    if (catMatch) {
      const path = catMatch[1].replace(/^\.\//, '')
      const content = mockFiles[path]
      if (content !== undefined) {
        inspectLines.push(`<ref tool="${shell.refName}">\n<stdout>${content}</stdout>\n<stderr></stderr>\n<exitCode>0</exitCode>\n</ref>`)
        continue
      }
    }
    // Generic success for everything else
    inspectLines.push(`<ref tool="${shell.refName}">\n<stdout></stdout>\n<stderr></stderr>\n<exitCode>0</exitCode>\n</ref>`)
  }

  // --- fs-edit ---
  for (const fsEdit of parsed.fsEdits) {
    const content = mockFiles[fsEdit.path]
    if (content !== undefined && fsEdit.oldText && content.includes(fsEdit.oldText)) {
      mockFiles[fsEdit.path] = content.replace(fsEdit.oldText, fsEdit.newText)
      inspectLines.push(`<ref tool="fs-edit">Applied edit to ${fsEdit.path}</ref>`)
    } else if (content === undefined) {
      resultLines.push(`<tool name="fs-edit"><error>ENOENT: no such file or directory '${fsEdit.path}'</error></tool>`)
    } else {
      inspectLines.push(`<ref tool="fs-edit">Applied edit to ${fsEdit.path}</ref>`)
    }
  }

  // --- fs-write ---
  for (const fsWrite of parsed.fsWrites) {
    mockFiles[fsWrite.path] = fsWrite.content
    inspectLines.push(`<ref tool="fs-write" />`)
  }

  // --- artifact-read ---
  for (const artRead of parsed.artifactReads) {
    const content = artifactStore.get(artRead.id)
    if (content !== undefined) {
      inspectLines.push(`<ref tool="${artRead.refName}">${content}</ref>`)
    } else {
      resultLines.push(`<tool name="artifact-read"><error>Artifact "${artRead.id}" does not exist</error></tool>`)
    }
  }

  // --- agent-dismiss ---
  for (const _dismiss of parsed.agentDismisses) {
    // Silent success — no output needed
  }

  return { resultLines, inspectLines }
}

/**
 * Build the results + agent responses injected after an orchestrator turn.
 *
 * Handles:
 * - Agent responses with artifacts (agents create their own artifacts)
 * - Direct tool results (fs-read, fs-search, shell, etc.) with inspect refs
 * - Agents status block
 */
function buildTurnFeedback(
  parsed: ParsedOrchestratorResponse,
  scenario: DispatchScenario,
  artifactStore: Map<string, string>,
): string {
  const parts: string[] = []
  const mockFiles = scenario.mockFiles ?? {}

  // Build results block
  const { resultLines, inspectLines } = buildToolResults(parsed, mockFiles, artifactStore)
  const allResultLines = ['<results>', ...resultLines]
  if (inspectLines.length > 0) {
    allResultLines.push('<inspect>', ...inspectLines, '</inspect>')
  }
  allResultLines.push('</results>')
  parts.push(allResultLines.join('\n'))

  // Agent responses for each created agent
  for (const ac of parsed.agentCreates) {
    const agentId = ac.agentId || nextAgentId(ac.type)
    const syntheticResponse = scenario.syntheticResponses?.[ac.type]

    if (syntheticResponse) {
      parts.push(`<agent_response from="${agentId}">\n${syntheticResponse.message}\n</agent_response>`)

      if (syntheticResponse.artifactContent) {
        const artifactId = `${agentId}-report`
        artifactStore.set(artifactId, syntheticResponse.artifactContent)

        for (const wId of ac.writableArtifactIds) {
          artifactStore.set(wId, syntheticResponse.artifactContent)
        }

        parts.push(`<artifact id="${artifactId}">\n${syntheticResponse.artifactContent}\n</artifact>`)
      }
    } else {
      parts.push(`<agent_response from="${agentId}">\nDone.\n</agent_response>`)
    }
  }

  // Agents status — all deployed agents are now idle
  if (parsed.agentCreates.length > 0) {
    const statusLines = parsed.agentCreates.map(ac => {
      const agentId = ac.agentId || `${ac.type}-1`
      return `- ${agentId} (${ac.type}): idle`
    })
    parts.push(`<agents_status>\n${statusLines.join('\n')}\n</agents_status>`)
  }

  return parts.join('\n')
}

/**
 * Build results for a turn with only direct tool uses (no agent creates).
 */
function buildDirectToolFeedback(
  parsed: ParsedOrchestratorResponse,
  scenario: DispatchScenario,
  artifactStore: Map<string, string>,
): string {
  const mockFiles = scenario.mockFiles ?? {}
  const { resultLines, inspectLines } = buildToolResults(parsed, mockFiles, artifactStore)

  const allLines = ['<results>', ...resultLines]
  if (inspectLines.length > 0) {
    allLines.push('<inspect>', ...inspectLines, '</inspect>')
  }
  allLines.push('</results>')
  return allLines.join('\n')
}

/**
 * Build the user-role message injected after a proposal.
 * Simulates user approval or rejection.
 */
function buildProposalFeedback(scenario: DispatchScenario): string {
  const result = scenario.proposalResult!
  const parts: string[] = []

  parts.push('<results>\n</results>')

  if (result.approved) {
    parts.push(`<autonomous_period_started taskId="eval-task-1">
The user has approved this task. You are now in an autonomous period.
Keep working until ALL criteria are met. Do not message the user unless absolutely necessary.
Criteria will be independently verified before the autonomous period ends.
</autonomous_period_started>`)
  } else if (result.feedback) {
    parts.push(`<user_task_feedback taskId="eval-task-1">\n${result.feedback}\n</user_task_feedback>`)
  }

  return parts.join('\n')
}

// =============================================================================
// Turn loop
// =============================================================================

const SAFETY_CAP = 20

async function runTurnLoop(
  scenario: DispatchScenario,
  modelSpec: ModelSpec,
): Promise<{ allResponses: string[]; error?: string }> {
  const messages: ChatMessage[] = [...scenario.messages]
  const allResponses: string[] = []
  const artifactStore = new Map<string, string>()

  agentIdCounter = 0

  for (let turn = 0; turn < SAFETY_CAP; turn++) {
    let response: string
    try {
      response = await callModel(getSystemPrompt(), messages, modelSpec)
    } catch (error) {
      return { allResponses, error: `Model call failed on turn ${turn + 1}: ${error instanceof Error ? error.message : String(error)}` }
    }

    allResponses.push(response)
    const parsed = parseOrchestratorResponse(response)

    // Track artifact creates
    for (const ac of parsed.artifactCreates) {
      artifactStore.set(ac.id, '')
    }

    // Add assistant turn to conversation
    messages.push({ role: 'assistant', content: [response] })

    // Terminal: proposal made
    if (parsed.proposes.length > 0) {
      if (scenario.proposalResult) {
        const feedback = buildProposalFeedback(scenario)
        messages.push({ role: 'user', content: [feedback] })
        scenario.proposalResult = undefined
        continue
      }
      break
    }

    // Terminal: submitted
    if (parsed.submits.length > 0) break

    // Check if anything actionable happened
    const hasAgents = parsed.agentCreates.length > 0
    const hasReads = parsed.fsReads.length > 0 || parsed.artifactReads.length > 0
    const hasSearches = parsed.fsSearches.length > 0
    const hasTrees = parsed.fsTrees.length > 0
    const hasShells = parsed.shells.length > 0
    const hasEdits = parsed.fsEdits.length > 0
    const hasWrites = parsed.fsWrites.length > 0
    const hasDismisses = parsed.agentDismisses.length > 0
    const hasArtifactCreates = parsed.artifactCreates.length > 0
    const hasArtifactWrites = parsed.artifactWrites.length > 0
    const hasAnyAction = hasAgents || hasReads || hasSearches || hasTrees || hasShells ||
      hasEdits || hasWrites || hasDismisses || hasArtifactCreates || hasArtifactWrites

    if (!hasAnyAction) break

    // Process artifact writes into the store
    for (const aw of parsed.artifactWrites) {
      artifactStore.set(aw.id, aw.content)
    }

    // Build feedback
    let feedback: string
    if (hasAgents) {
      feedback = buildTurnFeedback(parsed, scenario, artifactStore)
    } else {
      feedback = buildDirectToolFeedback(parsed, scenario, artifactStore)
    }
    messages.push({ role: 'user', content: [feedback] })
  }

  return { allResponses }
}

// =============================================================================
// Execution
// =============================================================================

function makeFail(scenarioId: string, message: string, checks: readonly { id: string }[]): ScenarioResult {
  const checkResults: Record<string, CheckResult> = {}
  for (const check of checks) {
    checkResults[check.id] = { passed: false, message }
  }
  return { scenarioId, checks: checkResults, passed: false, score: 0, rawResponse: '' }
}

async function executeScenario(scenario: DispatchScenario, modelSpec: ModelSpec): Promise<ScenarioResult> {
  // Deep copy mockFiles so edits/writes don't mutate the original scenario
  const scenarioCopy = {
    ...scenario,
    proposalResult: scenario.proposalResult ? { ...scenario.proposalResult } : undefined,
    mockFiles: scenario.mockFiles ? { ...scenario.mockFiles } : undefined,
  }

  const { allResponses, error } = await runTurnLoop(scenarioCopy, modelSpec)

  if (error && allResponses.length === 0) {
    return makeFail(scenario.id, error, scenario.checks)
  }

  const rawResponse = allResponses.join('\n\n---TURN---\n\n')

  const checks: Record<string, CheckResult> = {}
  let allPassed = true
  for (const check of scenario.checks) {
    const result = check.evaluate(rawResponse, null as never)
    checks[check.id] = result
    if (!result.passed) allPassed = false
  }

  const scores = Object.values(checks)
  const avgScore = scores.length > 0
    ? scores.reduce((sum, c) => sum + (c.score ?? (c.passed ? 1 : 0)), 0) / scores.length
    : 0

  return {
    scenarioId: scenario.id,
    checks,
    passed: allPassed,
    score: avgScore,
    rawResponse,
  }
}

// =============================================================================
// Variants
// =============================================================================

const VARIANTS: EvalVariant[] = [
  { id: 'quick-fix', label: 'Quick Fix', count: 2 },
  { id: 'feature', label: 'Feature', count: 3 },
  { id: 'bugfix', label: 'Bug Fix', count: 2 },
  { id: 'research', label: 'Research', count: 1 },
  { id: 'trivial', label: 'Trivial', count: 1 },
  { id: 'lifecycle', label: 'Lifecycle', count: 3 },
]

// =============================================================================
// Export
// =============================================================================

export const orchestratorDispatchEval: RunnableEval = {
  id: 'orchestrator-dispatch',
  name: 'Orchestrator Dispatch Behavior',
  description: `Tests orchestrator subagent dispatching across ${ALL_SCENARIOS.length} task patterns`,
  scenarios: ALL_SCENARIOS,
  variants: VARIANTS,
  defaultConcurrency: 4,

  async runScenario(scenario: Scenario, modelSpec: ModelSpec): Promise<ScenarioResult> {
    return executeScenario(scenario as DispatchScenario, modelSpec)
  },
}
