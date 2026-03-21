import type { DisplayMessage, ThinkBlockStep } from '@magnitudedev/agent'
import { displayBindingRegistry } from '../visuals/registry'

function normalize(text: string): string {
  return text.trim().replace(/\s+/g, ' ')
}

function getToolLiveText(step: ThinkBlockStep): string | null {
  if (step.type !== 'tool') return null
  if (step.toolKey && typeof step.visualState === 'object' && step.visualState !== null) {
    const binding = displayBindingRegistry.getAny(step.toolKey)
    const value = binding?.summary(step.visualState as { state: object; streaming: unknown })
    if (typeof value === 'string') {
      const normalized = normalize(value)
      if (normalized.length > 0) return normalized
    }
  }
  if (typeof step.label === 'string') {
    const normalized = normalize(step.label)
    if (normalized.length > 0) return normalized
  }
  return null
}

export function selectLatestLiveActivityFromThinkSteps(
  steps: readonly ThinkBlockStep[],
): string | null {
  for (let i = steps.length - 1; i >= 0; i--) {
    const step = steps[i]
    if (step.type === 'tool') {
      const text = getToolLiveText(step)
      if (text) return text
      continue
    }
    if (step.type === 'communication') {
      const text = normalize(step.preview)
      if (text.length > 0) return text
      const fallback = normalize(step.content)
      if (fallback.length > 0) return fallback
      continue
    }
    if (step.type === 'thinking' && typeof step.content === 'string') {
      const text = normalize(step.content)
      if (text.length > 0) return text
    }
  }
  return null
}

function getMessageLiveText(msg: DisplayMessage): string | null {
  if (msg.type === 'agent_communication') {
    const text = normalize(msg.preview)
    return text.length > 0 ? text : null
  }

  if (msg.type === 'think_block') {
    return selectLatestLiveActivityFromThinkSteps(msg.steps)
  }

  return null
}

export function selectLatestLiveActivityFromMessages(
  messages: readonly DisplayMessage[],
): string | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const text = getMessageLiveText(messages[i])
    if (text) return text
  }
  return null
}

export function selectLatestLiveActivityForSubagentTab(
  messages: readonly DisplayMessage[],
): string | null {
  return selectLatestLiveActivityFromMessages(messages)
}