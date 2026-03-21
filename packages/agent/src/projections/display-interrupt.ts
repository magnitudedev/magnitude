export interface InterruptResultLike {
  readonly status: string
}

export interface InterruptToolStepLike {
  readonly id?: string
  readonly type: 'tool'
  readonly toolKey?: string
  readonly result?: InterruptResultLike
  readonly visualState?: unknown
}

export function finalizeOpenToolStepsAsInterruptedInSteps<TStep extends { readonly type: string }>(
  steps: readonly TStep[],
  reduceVisual: (toolKey: string | undefined, visualState: unknown, stepId: string | undefined) => unknown
): readonly TStep[] {
  return steps.map((step): TStep => {
    if (step.type !== 'tool') return step

    const toolStep = step as TStep & InterruptToolStepLike
    if (toolStep.result) return step

    const toolKey = typeof toolStep.toolKey === 'string' ? toolStep.toolKey : undefined
    const stepId = typeof toolStep.id === 'string' ? toolStep.id : undefined
    return {
      ...toolStep,
      result: { status: 'interrupted' as const },
      visualState: reduceVisual(toolKey, toolStep.visualState, stepId),
    } as TStep
  })
}