import { useCallback, useMemo } from "react"
import { Atom, Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Effect } from "effect"
import { LocalInferenceHardwareMirror, type ModelServingConfigurationId } from "@magnitudedev/sdk"
import { OnboardingModelSetup } from "../local-models/setup"
import { useAgentClient } from "../state/agent-client-context"
import { useMirroredStateAtom } from "./use-mirrored-state"

export const useOnboardingModelSetup = () => {
  const client = useAgentClient()
  const hardwareAtom = useMirroredStateAtom(LocalInferenceHardwareMirror)
  const service = useMemo(() => client.effectQuery.runtime.atom(OnboardingModelSetup), [client])
  const state = useMemo(() => Atom.make((get) =>
    Result.flatMap(get(service), (setup) => get(setup.state))), [service])
  const startAction = useMemo(() => Atom.keepAlive(client.effectQuery.runtime.fn<ModelServingConfigurationId>()(
    (configurationId) => Effect.flatMap(
      OnboardingModelSetup,
      (setup) => setup.start(configurationId),
    ),
    { concurrent: true },
  )), [client])
  const cancelAction = useMemo(() => Atom.keepAlive(client.effectQuery.runtime.fn(
    () => Effect.flatMap(OnboardingModelSetup, (setup) => setup.cancel),
    { concurrent: true },
  )), [client])
  const skipAction = useMemo(() => Atom.keepAlive(client.effectQuery.runtime.fn(
    () => Effect.flatMap(OnboardingModelSetup, (setup) => setup.skip),
    { concurrent: true },
  )), [client])
  const setup = useAtomSet(startAction)
  const cancel = useAtomSet(cancelAction)
  const skip = useAtomSet(skipAction)
  return {
    hardware: Result.map(useAtomValue(hardwareAtom), ({ state }) => state),
    state: useAtomValue(state),
    startResult: useAtomValue(startAction),
    cancelResult: useAtomValue(cancelAction),
    skipResult: useAtomValue(skipAction),
    setup: useCallback((configurationId: ModelServingConfigurationId) => {
      setup(configurationId)
    }, [setup]),
    cancel: useCallback(() => cancel(), [cancel]),
    skip: useCallback(() => skip(), [skip]),
  }
}
