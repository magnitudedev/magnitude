import type { DisplayRootStatus } from '@magnitudedev/sdk'

type DisplayRootDetail = Extract<
  DisplayRootStatus,
  { readonly _tag: 'Working' }
>['detail']

export function formatTokenCount(tokens: number): string {
  const count = Math.max(0, Math.floor(tokens))
  return count < 1_000 ? String(count) : `${(count / 1_000).toFixed(1)}k`
}

export interface RootDetailSegments {
  readonly keyword: 'Waiting for model' | 'Prefill' | 'Thinking' | null
  readonly detail: string | null
  readonly trailing: string | null
}

export function rootDetailSegments(detail: DisplayRootDetail): RootDetailSegments {
  switch (detail._tag) {
    case 'NoDetail':
      return { keyword: null, detail: null, trailing: null }
    case 'WaitingForModel':
      return { keyword: 'Waiting for model', detail: null, trailing: null }
    case 'Thinking':
      return { keyword: 'Thinking', detail: null, trailing: null }
    case 'Prefill': {
      const totalTokens = Math.max(0, Math.floor(detail.totalTokens))
      const cachedTokens = Math.min(
        Math.max(0, Math.floor(detail.cachedTokens)),
        totalTokens,
      )
      const completedTokens = Math.min(
        Math.max(0, Math.floor(detail.completedTokens)),
        totalTokens,
      )
      const effectiveTotal = totalTokens - cachedTokens
      const effectiveCompleted = Math.max(0, completedTokens - cachedTokens)
      return {
        keyword: 'Prefill',
        detail: effectiveTotal > 0
          ? `${formatTokenCount(effectiveCompleted)} / ${formatTokenCount(effectiveTotal)} tokens`
          : null,
        trailing: cachedTokens > 0 ? `${formatTokenCount(cachedTokens)} cached` : null,
      }
    }
  }
}
