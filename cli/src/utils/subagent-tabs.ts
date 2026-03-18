import type { DisplayMessage, ForkActivityMessage } from '@magnitudedev/agent'

export function sumForkToolCounts(toolCounts: ForkActivityMessage['toolCounts']): number {
  return Object.values(toolCounts).reduce((sum, n) => sum + n, 0)
}

function latestThinkToolStepLabel(messages: readonly DisplayMessage[]): string | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i]
    if (msg.type !== 'think_block') continue
    for (let j = msg.steps.length - 1; j >= 0; j--) {
      const step = msg.steps[j]
      if (step?.type === 'tool' && typeof step.label === 'string') {
        const text = step.label.trim().replace(/\s+/g, ' ')
        if (text.length > 0) return text
      }
    }
  }
  return null
}

function latestThinkingSnippet(messages: readonly DisplayMessage[]): string | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i]
    if (msg.type !== 'think_block') continue
    for (let j = msg.steps.length - 1; j >= 0; j--) {
      const step = msg.steps[j]
      if (step?.type === 'thinking' && typeof step.content === 'string') {
        const text = step.content.trim().replace(/\s+/g, ' ')
        if (text.length > 0) return text
      }
    }
  }
  return null
}

function latestAgentCommunicationPreview(messages: readonly DisplayMessage[]): string | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i]
    if (msg.type === 'agent_communication') {
      const text = msg.preview.trim()
      if (text.length > 0) return text
    }
  }
  return null
}

export function deriveSubagentStatusLine(messages: readonly DisplayMessage[]): string {
  return (
    latestThinkToolStepLabel(messages)
    ?? latestThinkingSnippet(messages)
    ?? latestAgentCommunicationPreview(messages)
    ?? 'Running…'
  )
}

export function truncateSubagentTabText(text: string, max = 44): string {
  const normalized = text.trim().replace(/\s+/g, ' ')
  if (normalized.length <= max) return normalized
  return normalized.slice(0, Math.max(0, max - 1)) + '…'
}