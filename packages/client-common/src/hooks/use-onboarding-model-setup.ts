import { useCallback, useMemo } from "react"
import { Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { LocalInferenceHardwareMirror, type ModelServingConfigurationId } from "@magnitudedev/sdk"
import { onboardingModelSetupService } from "../local-models/setup"
import { useAgentClient } from "../state/agent-client-context"
import { useMirroredStateAtom } from "./use-mirrored-state"

export const useOnboardingModelSetup = () => {
  const client = useAgentClient()
  const hardwareAtom = useMirroredStateAtom(LocalInferenceHardwareMirror)
  const service = useMemo(() => onboardingModelSetupService(client), [client])
  const setup = useAtomSet(service.start)
  const cancel = useAtomSet(service.cancel)
  const skip = useAtomSet(service.skip)
  return {
    hardware: Result.map(useAtomValue(hardwareAtom), ({ state }) => state),
    state: useAtomValue(service.state),
    startResult: useAtomValue(service.start),
    cancelResult: useAtomValue(service.cancel),
    skipResult: useAtomValue(service.skip),
    setup: useCallback((configurationId: ModelServingConfigurationId) => {
      setup(configurationId)
    }, [setup]),
    cancel: useCallback(() => cancel(), [cancel]),
    skip: useCallback(() => skip(), [skip]),
  }
}
