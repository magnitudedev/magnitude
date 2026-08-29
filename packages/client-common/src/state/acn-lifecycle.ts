import { useMemo } from "react"
import {
  Atom,
  Result,
  useAtomSet,
  useAtomValue,
} from "@effect-atom/atom-react"
import { Option } from "effect"
import type { AcnLifecycleState } from "@magnitudedev/sdk"
import { useAcnStartup } from "./acn-startup"

export function useAcnLifecycle(
  initialState: AcnLifecycleState,
): {
  readonly state: AcnLifecycleState
  readonly retry: () => void
} {
  const startup = useAcnStartup()
  const stateAtom = useMemo(
    () => Atom.make(startup.state.changes, { initialValue: initialState }),
    [startup, initialState],
  )
  const retryAtom = useMemo(
    () => Atom.fn<"RetryAcn">()(() => startup.retry),
    [startup],
  )
  const state = Option.getOrElse(
    Result.value(useAtomValue(stateAtom)),
    () => initialState,
  )
  const runRetry = useAtomSet(retryAtom)

  return {
    state,
    retry: () => runRetry("RetryAcn"),
  }
}
