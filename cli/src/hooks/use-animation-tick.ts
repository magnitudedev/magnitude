import { useSyncExternalStore } from "react"
import {
  getAnimationTickFrozenSnapshot,
  getAnimationTickSnapshot,
  subscribeAnimationNoop,
  subscribeAnimationTick,
} from "@magnitudedev/client-common"

export const useAnimationTick = (active: boolean): number => {
  const snapshot = active
    ? getAnimationTickSnapshot
    : getAnimationTickFrozenSnapshot
  return useSyncExternalStore(
    active ? subscribeAnimationTick : subscribeAnimationNoop,
    snapshot,
    snapshot,
  )
}
