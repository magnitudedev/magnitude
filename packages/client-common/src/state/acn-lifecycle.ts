import { useMemo } from "react"
import {
  Atom,
  Result,
  useAtomSet,
  useAtomValue,
} from "@effect-atom/atom-react"
import { Option } from "effect"
import type { AcnLifecycleState } from "@magnitudedev/sdk"
import { usePlatform } from "../platform/platform-context"

export function useAcnLifecycle(
  initialState: AcnLifecycleState,
): {
  readonly state: AcnLifecycleState
  readonly retry: () => void
} {
  const { acnStartup } = usePlatform()
  const stateAtom = useMemo(
    () =>
      Atom.make(acnStartup.state.changes, {
        initialValue: initialState,
      }),
    [acnStartup, initialState],
  )
  const retryAtom = useMemo(
    () => Atom.fn<"RetryAcn">()(() => acnStartup.retry),
    [acnStartup],
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
