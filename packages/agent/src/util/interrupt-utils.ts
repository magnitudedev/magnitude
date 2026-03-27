import { Effect } from 'effect'
import { observeOutput, type ReactorState } from '@magnitudedev/xml-act'
import type { ObservedResult, ResponsePart, ToolResult, TurnCompleted, TurnToolCall } from '../events'
import { mapXmlToolResult } from './tool-result'
import { getAgentDefinition, type AgentVariant } from '../agents'
import { isToolKey, type ToolKey, type AgentCatalogEntry } from '../catalog'
import { CanonicalTurnProjection, type CanonicalTurnState } from '../projections/canonical-turn'
import { AgentStatusProjection, getAgentByForkId } from '../projections/agent-status'
import { ReplayProjection } from '../projections/replay'

function buildResponseParts(canonical: CanonicalTurnState): readonly ResponsePart[] {
  const textChunks: string[] = []

  for (const lens of canonical.lenses ?? []) {
    if (!lens.content) continue
    textChunks.push(lens.content)
  }

  for (const message of canonical.messages) {
    if (!message.text) continue
    textChunks.push(message.text)
  }

  const responseParts: ResponsePart[] = canonical.thinkBlocks
    .filter(block => block.content.length > 0)
    .map(block => ({ type: 'thinking', content: block.content }))

  const textContent = textChunks.join('')
  if (textContent.length > 0) {
    responseParts.unshift({ type: 'text', content: textContent })
  }

  return responseParts
}

function buildObservedResults(replay: ReactorState): readonly ObservedResult[] {
  const observedResults: ObservedResult[] = []

  for (const [toolCallId, outcome] of replay.toolOutcomes.entries()) {
    if (outcome._tag !== 'Completed') continue
    if (outcome.result._tag !== 'Success') continue

    observedResults.push({
      toolCallId,
      tagName: outcome.result.outputTree.tag,
      query: outcome.result.query,
      content: observeOutput(outcome.result.outputTree.tree, outcome.result.query),
    })
  }

  return observedResults
}

export const buildInterruptedTurnCompleted = (params: {
  forkId: string | null
  turnId: string
  chainId: string | null
}) => Effect.gen(function* () {
  const { forkId, turnId, chainId } = params

  const canonicalProjection = yield* CanonicalTurnProjection.Tag
  const replayProjection = yield* ReplayProjection.Tag
  const agentProjection = yield* AgentStatusProjection.Tag

  const canonical = yield* canonicalProjection.getFork(forkId)
  const replay = yield* replayProjection.getFork(forkId)
  const agentState = yield* agentProjection.get

  const variant: AgentVariant = forkId
    ? ((getAgentByForkId(agentState, forkId)?.role ?? 'builder') as AgentVariant)
    : 'lead'

  const agentDef = getAgentDefinition(variant)
  const tagToMeta = new Map<string, { toolKey: ToolKey; group: string; toolName: string }>()
  for (const toolKey of agentDef.tools.keys) {
    const entry = agentDef.tools.entries[toolKey] as AgentCatalogEntry
    const tool = entry.tool
    const tagName = entry.binding.toXmlTagBinding().tag
    if (isToolKey(toolKey)) {
      tagToMeta.set(tagName, {
        toolKey,
        group: tool.group ?? 'default',
        toolName: tool.name,
      })
    }
  }

  const toolCalls: TurnToolCall[] = []
  for (const [toolCallId, tagName] of replay.toolCallMap.entries()) {
    const meta = tagToMeta.get(tagName)
    if (!meta) continue

    const outcome = replay.toolOutcomes.get(toolCallId)
    const result: ToolResult = outcome
      ? outcome._tag === 'Completed'
        ? mapXmlToolResult(outcome.result)
        : { status: 'error', message: 'Tool input parse error' }
      : { status: 'interrupted' }

    toolCalls.push({
      toolKey: meta.toolKey,
      group: meta.group,
      toolName: meta.toolName,
      result,
    })
  }

  const responseParts = buildResponseParts(canonical)
  const observedResults = buildObservedResults(replay)

  const event: TurnCompleted = {
    type: 'turn_completed',
    forkId,
    turnId,
    chainId: chainId ?? '',
    strategyId: 'xml-act',
    responseParts,
    toolCalls,
    observedResults,
    result: { success: false, error: 'Interrupted', cancelled: true },
    inputTokens: null,
    outputTokens: null,
    cacheReadTokens: null,
    cacheWriteTokens: null,
    providerId: null,
    modelId: null,
  }

  return event
})