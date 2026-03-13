import { Effect } from 'effect'
import { outputToText, type OutputNode, type ReactorState, type XmlToolResult } from '@magnitudedev/xml-act'
import type { InspectResult, ResponsePart, ToolResult, TurnCompleted, TurnToolCall } from '../events'
import { mapXmlToolResult } from './tool-result'
import { getAgentDefinition, type AgentVariant } from '../agents'
import { defaultXmlTagName } from '../tools'
import { CanonicalTurnProjection, type CanonicalTurnState } from '../projections/canonical-turn'
import { AgentProjection, getAgentByForkId } from '../projections/agent'
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

function buildInspectResults(refStore: ReactorState['refStore']): readonly InspectResult[] {
  const inspectResults: InspectResult[] = []

  for (const [tag, trees] of refStore.entries()) {
    trees.forEach((tree: OutputNode, index) => {
      const recency = trees.length - index - 1
      const toolRef = recency === 0 ? tag : `${tag}~${recency}`
      inspectResults.push({
        status: 'resolved',
        toolRef,
        content: outputToText(tree),
      })
    })
  }

  return inspectResults
}

export const buildInterruptedTurnCompleted = (params: {
  forkId: string | null
  turnId: string
  chainId: string | null
}) => Effect.gen(function* () {
  const { forkId, turnId, chainId } = params

  const canonicalProjection = yield* CanonicalTurnProjection.Tag
  const replayProjection = yield* ReplayProjection.Tag
  const agentProjection = yield* AgentProjection.Tag

  const canonical = yield* canonicalProjection.getFork(forkId)
  const replay = yield* replayProjection.getFork(forkId)
  const agentState = yield* agentProjection.get

  const variant: AgentVariant = forkId
    ? ((getAgentByForkId(agentState, forkId)?.role ?? 'builder') as AgentVariant)
    : 'orchestrator'

  const agentDef = getAgentDefinition(variant)
  const tagToMeta = new Map<string, { toolKey: string; group: string; toolName: string }>()
  for (const [toolKey, tool] of Object.entries(agentDef.tools)) {
    if (!tool) continue
    const concreteTool = tool as { name: string; group?: string }
    const tagName = defaultXmlTagName(concreteTool as any)
    tagToMeta.set(tagName, {
      toolKey,
      group: concreteTool.group ?? 'default',
      toolName: concreteTool.name,
    })
  }

  const toolCalls: TurnToolCall[] = []
  for (const [toolCallId, tagName] of replay.toolCallMap.entries()) {
    const meta = tagToMeta.get(tagName) ?? {
      toolKey: tagName,
      group: tagName,
      toolName: tagName,
    }

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
  const inspectResults = buildInspectResults(replay.refStore)

  const event: TurnCompleted = {
    type: 'turn_completed',
    forkId,
    turnId,
    chainId: chainId ?? '',
    strategyId: 'xml-act',
    responseParts,
    toolCalls,
    inspectResults,
    result: { success: false, error: 'Interrupted', cancelled: true },
    inputTokens: null,
    outputTokens: null,
    cacheReadTokens: null,
    cacheWriteTokens: null,
  }

  return event
})