import { useMemo } from "react"
import {
  Atom,
  Result,
  useAtomSet,
  useAtomValue,
} from "@effect-atom/atom-react"
import { Option } from "effect"
import type { ServiceLifecycleState } from "../connection/lifecycle"
import { useServiceStartup } from "./service-startup"

export function useServiceLifecycle(
  initialState: ServiceLifecycleState,
): {
  readonly state: ServiceLifecycleState
  readonly retry: () => void
} {
  const startup = useServiceStartup()
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
