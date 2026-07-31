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
  const segments = modelRequestProgressSegments(activity)
  return {
    primary: segments.detail === null
      ? segments.label
      : `${segments.label} · ${segments.detail}`,
    secondary: segments.trailing,
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
      label: 'Prefilling',
      detail: `${formatTokenCount(newCompleted)} / ${formatTokenCount(newTotal)} input tokens`,
      trailing: `Using ${formatTokenCount(cachedTokens)} cached tokens`,
    }
  }
  return {
    label: 'Prefilling',
    detail: `${formatTokenCount(completedTokens)} / ${formatTokenCount(totalTokens)} input tokens`,
    trailing: null,
  }
}
