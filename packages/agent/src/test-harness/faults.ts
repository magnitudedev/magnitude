export type FaultScope =
  | 'persistence.persistNewEvents'
  | 'persistence.loadEvents'
  | `tool.execute:${string}`

export interface FaultPlan {
  scope: FaultScope
  when?: { count?: number }
  action:
    | { type: 'throw'; error: string | Error }
    | { type: 'delay'; ms: number }
}

export interface FaultRegistry {
  set(plan: FaultPlan): void
  clear(scope?: FaultScope): void
  check(scope: FaultScope): void
  checkAsync(scope: FaultScope): Promise<void>
}

type RuntimeFaultPlan = FaultPlan & {
  remainingCount: number | null
}

function toError(error: string | Error): Error {
  return error instanceof Error ? error : new Error(error)
}

function shouldApply(plan: RuntimeFaultPlan): boolean {
  if (plan.remainingCount === null) return true
  if (plan.remainingCount <= 0) return false
  plan.remainingCount -= 1
  return true
}

export function createFaultRegistry(): FaultRegistry {
  const plans = new Map<FaultScope, RuntimeFaultPlan>()

  const resolvePlan = (scope: FaultScope): RuntimeFaultPlan | undefined => {
    const plan = plans.get(scope)
    if (!plan) return undefined
    if (!shouldApply(plan)) {
      plans.delete(scope)
      return undefined
    }
    if (plan.remainingCount === 0) {
      plans.delete(scope)
    }
    return plan
  }

  return {
    set(plan) {
      plans.set(plan.scope, {
        ...plan,
        remainingCount: plan.when?.count ?? null,
      })
    },

    clear(scope) {
      if (scope) {
        plans.delete(scope)
      } else {
        plans.clear()
      }
    },

    check(scope) {
      const plan = resolvePlan(scope)
      if (!plan) return

      if (plan.action.type === 'throw') {
        throw toError(plan.action.error)
      }
    },

    async checkAsync(scope) {
      const plan = resolvePlan(scope)
      if (!plan) return

      const action = plan.action
      if (action.type === 'throw') {
        throw toError(action.error)
      } else if (action.ms > 0) {
        const ms = action.ms
        await new Promise<void>((resolve) => setTimeout(resolve, ms))
      }
    },
  }
}