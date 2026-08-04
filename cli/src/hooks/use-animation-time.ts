import { useSyncExternalStore } from 'react'
import {
  animationStep,
  getAnimationTimeFrozenSnapshot,
  getAnimationTimeSnapshot,
  subscribeAnimationClock,
  subscribeAnimationNoop,
} from '@magnitudedev/client-common'

export const useAnimationTime = (active: boolean): number => useSyncExternalStore(
  active ? subscribeAnimationClock : subscribeAnimationNoop,
  active ? getAnimationTimeSnapshot : getAnimationTimeFrozenSnapshot,
  active ? getAnimationTimeSnapshot : getAnimationTimeFrozenSnapshot,
)

const stepSnapshots = new Map<number, () => number>()

function stepSnapshot(stepDurationMs: number): () => number {
  const existing = stepSnapshots.get(stepDurationMs)
  if (existing !== undefined) return existing
  const snapshot = () => animationStep(getAnimationTimeSnapshot(), stepDurationMs)
  stepSnapshots.set(stepDurationMs, snapshot)
  return snapshot
}

export function useAnimationStep(active: boolean, stepDurationMs: number): number {
  const getStepSnapshot = stepSnapshot(stepDurationMs)
  return useSyncExternalStore(
    active ? subscribeAnimationClock : subscribeAnimationNoop,
    active ? getStepSnapshot : getAnimationTimeFrozenSnapshot,
    active ? getStepSnapshot : getAnimationTimeFrozenSnapshot,
  )
}
