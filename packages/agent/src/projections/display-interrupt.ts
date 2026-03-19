export interface InterruptResultLike {
  readonly status: string
}

export interface InterruptToolStepLike {
  readonly type: 'tool'
  readonly toolKey?: string
  readonly result?: InterruptResultLike
  readonly visualState?: unknown
}

export type InterruptStepLike = InterruptToolStepLike | { readonly type: string; readonly [key: string]: unknown }

export function finalizeOpenToolStepsAsInterruptedInSteps<TStep extends InterruptStepLike>(
  steps: readonly TStep[],
  reduceVisual: (toolKey: string | undefined, visualState: unknown) => unknown
): readonly TStep[] {
  return steps.map((step): TStep => {
    if (step.type !== 'tool' || step.result) return step

    return {
      ...step,
      result: { status: 'interrupted' as const },
      visualState: step.visualState !== undefined
        ? reduceVisual(step.toolKey, step.visualState)
        : step.visualState,
    } as TStep
  })
}