import type { DisplayModelRequestActivity } from '@magnitudedev/sdk'

export function formatTokenCount(tokens: number): string {
  const count = Math.max(0, Math.floor(tokens))
  return count < 1_000 ? String(count) : `${(count / 1_000).toFixed(1)}k`
}

export interface ModelRequestProgressCopy {
  readonly primary: string
  readonly secondary: string | null
}

export function modelRequestProgressCopy(
  activity: DisplayModelRequestActivity,
): ModelRequestProgressCopy {
  if (activity.phase === 'queued') {
    return { primary: 'Waiting for the local model', secondary: null }
  }
  if (activity.phase === 'preparing') {
    return { primary: 'Loading conversation into the model', secondary: null }
  }

  const completedTokens = activity.completedTokens ?? 0
  const totalTokens = activity.totalTokens ?? 0
  const cachedTokens = Math.min(activity.cachedTokens ?? 0, totalTokens)
  if (cachedTokens > 0) {
    const newCompleted = Math.max(0, completedTokens - cachedTokens)
    const newTotal = Math.max(0, totalTokens - cachedTokens)
    return {
      primary: `Adding new messages to the model · ${formatTokenCount(newCompleted)} / ${formatTokenCount(newTotal)} tokens`,
      secondary: `Reusing ${formatTokenCount(cachedTokens)} tokens from earlier messages`,
    }
  }

  return {
    primary: `Loading conversation into the model · ${formatTokenCount(completedTokens)} / ${formatTokenCount(totalTokens)} tokens`,
    secondary: null,
  }
}

export interface ModelRequestProgressSegments {
  readonly label: string
  readonly detail: string | null
  readonly trailing: string | null
}

export function modelRequestProgressSegments(
  activity: DisplayModelRequestActivity,
): ModelRequestProgressSegments {
  if (activity.phase === 'queued') {
    return { label: 'Waiting for the local model', detail: null, trailing: null }
  }
  if (activity.phase === 'preparing') {
    return { label: 'Loading conversation into the model', detail: null, trailing: null }
  }

  const completedTokens = activity.completedTokens ?? 0
  const totalTokens = activity.totalTokens ?? 0
  const cachedTokens = Math.min(activity.cachedTokens ?? 0, totalTokens)
  if (cachedTokens > 0) {
    const newCompleted = Math.max(0, completedTokens - cachedTokens)
    const newTotal = Math.max(0, totalTokens - cachedTokens)
    return {
      label: 'Adding new messages',
      detail: `${formatTokenCount(newCompleted)} / ${formatTokenCount(newTotal)} tokens`,
      trailing: `Reusing ${formatTokenCount(cachedTokens)} tokens`,
    }
  }
  return {
    label: 'Loading conversation into the model',
    detail: `${formatTokenCount(completedTokens)} / ${formatTokenCount(totalTokens)} tokens`,
    trailing: null,
  }
}
