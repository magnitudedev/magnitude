import { useCallback, useMemo } from "react"
import {
  Atom,
  Result,
  useAtomSet,
  useAtomValue,
} from "@effect-atom/atom-react"
import { Effect } from "effect"
import { LocalInferenceHardwareMirror, type ModelServingConfigurationId } from "@magnitudedev/sdk"
import { OnboardingModelSetup } from "../local-models/setup"
import { onboardingModelSetupViewAtom } from "../local-models/setup-view"
import { useAgentClient } from "../state/agent-client-context"
import { useMirroredStateAtom } from "./use-mirrored-state"

export const useOnboardingModelSetup = () => {
  const client = useAgentClient()
  const hardwareAtom = useMirroredStateAtom(LocalInferenceHardwareMirror)
  const view = useMemo(() => onboardingModelSetupViewAtom(client), [client])
  const retryAction = useMemo(() => client.effectQuery.runtime.fn(
    () => Effect.flatMap(OnboardingModelSetup, (setup) => setup.retry),
    { concurrent: true },
  ), [client])
  const openAction = useMemo(() => client.effectQuery.runtime.fn(
    () => Effect.flatMap(OnboardingModelSetup, (setup) => setup.open),
    { concurrent: true },
  ), [client])
  const selectAction = useMemo(() => client.effectQuery.runtime.fn<ModelServingConfigurationId>()(
    (configurationId) => Effect.flatMap(
      OnboardingModelSetup,
      (setup) => setup.select(configurationId),
    ),
    { concurrent: true },
  ), [client])
  const cancelAction = useMemo(() => client.effectQuery.runtime.fn(
    () => Effect.flatMap(OnboardingModelSetup, (setup) => setup.cancel),
    { concurrent: true },
  ), [client])
  const exitAction = useMemo(() => client.effectQuery.runtime.fn(
    () => Effect.flatMap(OnboardingModelSetup, (setup) => setup.exit),
    { concurrent: true },
  ), [client])
  const retry = useAtomSet(retryAction)
  const open = useAtomSet(openAction)
  const select = useAtomSet(selectAction)
  const cancel = useAtomSet(cancelAction)
  const exit = useAtomSet(exitAction)

  return {
    hardware: Result.map(useAtomValue(hardwareAtom), ({ state }) => state),
    view: useAtomValue(view),
    retry: useCallback(() => retry(), [retry]),
    open: useCallback(() => open(), [open]),
    select: useCallback((configurationId: ModelServingConfigurationId) => {
      select(configurationId)
    }, [select]),
    cancel: useCallback(() => cancel(), [cancel]),
    exit: useCallback(() => exit(), [exit]),
  }
}
