/**
 * Orchestrator Dispatch Eval
 *
 * Tests orchestrator subagent dispatching across multiple turns.
 * Uses the real orchestrator system prompt, calls the model in a loop,
 * and injects synthetic agent responses to drive the orchestrator.
 */

import type { RunnableEval, Scenario, ScenarioResult, ModelSpec, CheckResult, EvalVariant } from '../../types'
import type { ChatMessage } from '@magnitudedev/llm-core'
import { callModel } from '../../runner'
import { getAgentDefinition, generateXmlActToolDocs } from '@magnitudedev/agent'
import { getXmlActProtocol } from '@magnitudedev/agent-definition'
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
    cachedSystemPrompt = `${getXmlActProtocol('user', agentDef.thinkingLenses.slice())}\n\n${agentDef.systemPrompt}\n\n## Tools\n\n${toolDocs}`
  }
  return cachedSystemPrompt
}

// =============================================================================
// Synthetic response formatting
// =============================================================================

let agentIdCounter = 0

function nextAgentId(type: string): string {
  return `${type}-${++agentIdCounter}`
}

function buildToolResults(
  parsed: ParsedOrchestratorResponse,
  mockFiles: Record<string, string>,
  artifactStore: Map<string, string>,
): { resultLines: string[]; inspectLines: string[] } {
  const resultLines: string[] = []
  const inspectLines: string[] = []

  // fs-read
  for (const fsRead of parsed.fsReads) {
    const content = mockFiles[fsRead.path]
    if (content !== undefined) {
      inspectLines.push(`<ref tool="${fsRead.refName}">${content}</ref>`)
    } else {
      resultLines.push(`<tool name="fs-read"><error>ENOENT: no such file or directory '${fsRead.path}'</error></tool>`)
    }
  }

  // fs-search
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
        if (pattern.test(lines[i])) items.push(`<item file="${filePath}">${i + 1}|${lines[i]}</item>`)
        pattern.lastIndex = 0
      }
    }

    if (items.length > 0) inspectLines.push(`<ref tool="${fsSearch.refName}">\n${items.join('\n')}\n</ref>`)
    else inspectLines.push(`<ref tool="${fsSearch.refName}">No matches found</ref>`)
  }

  // fs-tree
  for (const fsTree of parsed.fsTrees) {
    const dirs = new Set<string>()
    const entries: string[] = []
    const prefix = fsTree.path === '.' ? '' : fsTree.path.replace(/^\.\//, '')

    for (const filePath of Object.keys(mockFiles).sort()) {
      if (prefix && !filePath.startsWith(prefix)) continue
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

  // shell
  for (const shell of parsed.shells) {
    const catMatch = shell.command.match(/cat\s+(\S+)/)
    if (catMatch) {
      const path = catMatch[1].replace(/^\.\//, '')
      const content = mockFiles[path]
      if (content !== undefined) {
        inspectLines.push(`<ref tool="${shell.refName}">\n<stdout>${content}</stdout>\n<stderr></stderr>\n<exitCode>0</exitCode>\n</ref>`)
        continue
      }
    }
    inspectLines.push(`<ref tool="${shell.refName}">\n<stdout></stdout>\n<stderr></stderr>\n<exitCode>0</exitCode>\n</ref>`)
  }

  // fs-edit
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

  // fs-write
  for (const fsWrite of parsed.fsWrites) {
    mockFiles[fsWrite.path] = fsWrite.content
    inspectLines.push(`<ref tool="fs-write" />`)
  }

  // artifact-read
  for (const artRead of parsed.artifactReads) {
    const content = artifactStore.get(artRead.id)
    if (content !== undefined) {
      inspectLines.push(`<ref tool="${artRead.refName}">${content}</ref>`)
    } else {
      resultLines.push(`<tool name="artifact-read"><error>Artifact "${artRead.id}" does not exist</error></tool>`)
    }
  }

  return { resultLines, inspectLines }
}

function buildTurnFeedback(
  parsed: ParsedOrchestratorResponse,
  scenario: DispatchScenario,
  artifactStore: Map<string, string>,
): string {
  const parts: string[] = []
  const mockFiles = scenario.mockFiles ?? {}

  const { resultLines, inspectLines } = buildToolResults(parsed, mockFiles, artifactStore)
  const allResultLines = ['<results>', ...resultLines]
  if (inspectLines.length > 0) allResultLines.push('<inspect>', ...inspectLines, '</inspect>')
  allResultLines.push('</results>')
  parts.push(allResultLines.join('\n'))

  for (const ac of parsed.agentCreates) {
    const agentId = ac.agentId || nextAgentId(ac.type)
    const syntheticResponse = scenario.syntheticResponses?.[ac.type]

    if (syntheticResponse) {
      parts.push(`<agent_response from="${agentId}">\n${syntheticResponse.message}\n</agent_response>`)

      if (syntheticResponse.artifactContent) {
        const artifactId = `${agentId}-report`
        artifactStore.set(artifactId, syntheticResponse.artifactContent)

        for (const wId of ac.writableArtifactIds) artifactStore.set(wId, syntheticResponse.artifactContent)

        parts.push(`<artifact id="${artifactId}">\n${syntheticResponse.artifactContent}\n</artifact>`)
      }
    } else {
      parts.push(`<agent_response from="${agentId}">\nDone.\n</agent_response>`)
    }
  }

  if (parsed.agentCreates.length > 0) {
    const statusLines = parsed.agentCreates.map(ac => {
      const agentId = ac.agentId || `${ac.type}-1`
      return `- ${agentId} (${ac.type}): idle`
    })
    parts.push(`<agents_status>\n${statusLines.join('\n')}\n</agents_status>`)
  }

  return parts.join('\n')
}

function buildDirectToolFeedback(
  parsed: ParsedOrchestratorResponse,
  scenario: DispatchScenario,
  artifactStore: Map<string, string>,
): string {
  const mockFiles = scenario.mockFiles ?? {}
  const { resultLines, inspectLines } = buildToolResults(parsed, mockFiles, artifactStore)

  const allLines = ['<results>', ...resultLines]
  if (inspectLines.length > 0) allLines.push('<inspect>', ...inspectLines, '</inspect>')
  allLines.push('</results>')
  return allLines.join('\n')
}

interface ConversationScript {
  approval?: string
  rejection?: string
  injectAfter?: number | 'first-plan-message'
}

function getConversationScript(scenario: DispatchScenario): ConversationScript | undefined {
  return (scenario as DispatchScenario & { conversationScript?: ConversationScript }).conversationScript
}

function wrapScriptedUserMessage(text: string): string {
  return `<user mode="text" at="2026-Mar-04 10:00:00">\n${text}\n</user>`
}

function hasAction(parsed: ParsedOrchestratorResponse): boolean {
  return (
    parsed.agentCreates.length > 0 ||
    parsed.fsReads.length > 0 ||
    parsed.fsSearches.length > 0 ||
    parsed.fsTrees.length > 0 ||
    parsed.shells.length > 0 ||
    parsed.fsEdits.length > 0 ||
    parsed.fsWrites.length > 0 ||
    parsed.agentDismisses.length > 0 ||

    parsed.artifactCreates.length > 0 ||
    parsed.artifactWrites.length > 0 ||
    parsed.artifactReads.length > 0
  )
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

  const script = getConversationScript(scenario)
  let scriptInjected = false
  let builderSeen = false

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

    for (const ac of parsed.artifactCreates) artifactStore.set(ac.id, '')

    if (parsed.agentCreates.some(a => a.type === 'builder')) builderSeen = true

    messages.push({ role: 'assistant', content: [response] })

    const readyByTurn = typeof script?.injectAfter === 'number' ? turn + 1 >= script.injectAfter : false
    const readyByFirstPlan = (script?.injectAfter ?? 'first-plan-message') === 'first-plan-message' &&
      !builderSeen &&
      parsed.messages.some(msg => msg.to.toLowerCase() === 'user')

    if (script && !scriptInjected && (readyByTurn || readyByFirstPlan)) {
      const payload = script.rejection ?? script.approval
      if (payload) {
        messages.push({ role: 'user', content: [wrapScriptedUserMessage(payload)] })
        scriptInjected = true
        continue
      }
    }

    if (!hasAction(parsed)) break

    for (const aw of parsed.artifactWrites) artifactStore.set(aw.id, aw.content)

    const feedback = parsed.agentCreates.length > 0
      ? buildTurnFeedback(parsed, scenario, artifactStore)
      : buildDirectToolFeedback(parsed, scenario, artifactStore)
    messages.push({ role: 'user', content: [feedback] })

    const completion = (scenario as DispatchScenario & {
      completionExpectations?: { requireBuilder?: boolean; requireReviewer?: boolean }
    }).completionExpectations

    if (completion) {
      const parsedAll = parseOrchestratorResponse(allResponses.join('\n'))
      const hasBuilder = completion.requireBuilder ? parsedAll.agentCreates.some(a => a.type === 'builder') : true
      const hasReviewer = completion.requireReviewer ? parsedAll.agentCreates.some(a => a.type === 'reviewer') : true
      if (hasBuilder && hasReviewer && parsed.messages.some(msg => msg.to.toLowerCase() === 'user')) break
    }
  }

  return { allResponses }
}

// =============================================================================
// Execution
// =============================================================================

function makeFail(scenarioId: string, message: string, checks: readonly { id: string }[]): ScenarioResult {
  const checkResults: Record<string, CheckResult> = {}
  for (const check of checks) checkResults[check.id] = { passed: false, message }
  return { scenarioId, checks: checkResults, passed: false, score: 0, rawResponse: '' }
}

async function executeScenario(scenario: DispatchScenario, modelSpec: ModelSpec): Promise<ScenarioResult> {
  const scenarioCopy = {
    ...scenario,
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