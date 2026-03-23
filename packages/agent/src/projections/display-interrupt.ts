interface InterruptStateLike {
  readonly phase?: string
}

export interface InterruptToolStepLike {
  readonly id?: string
  readonly type: 'tool'
  readonly toolKey?: string
  readonly state?: InterruptStateLike
}

function isTerminalPhase(phase: string | undefined): boolean {
  return phase === 'completed'
    || phase === 'error'
    || phase === 'rejected'
    || phase === 'interrupted'
}

export function finalizeOpenToolStepsAsInterruptedInSteps<TStep extends { readonly type: string }>(
  steps: readonly TStep[],
  reduceVisual: (toolKey: string | undefined, state: unknown, stepId: string | undefined) => unknown
): readonly TStep[] {
  return steps.map((step): TStep => {
    if (step.type !== 'tool') return step

    const toolStep = step as TStep & InterruptToolStepLike
    if (isTerminalPhase(toolStep.state?.phase)) return step

    const toolKey = typeof toolStep.toolKey === 'string' ? toolStep.toolKey : undefined
    const stepId = typeof toolStep.id === 'string' ? toolStep.id : undefined
    const nextState = reduceVisual(toolKey, toolStep.state, stepId)

    if (nextState && typeof nextState === 'object' && 'phase' in nextState) {
      const phase = nextState.phase
      if (typeof phase === 'string' && phase === 'interrupted') {
        return { ...toolStep, state: nextState } as TStep
      }
    }

    return step
  })
}