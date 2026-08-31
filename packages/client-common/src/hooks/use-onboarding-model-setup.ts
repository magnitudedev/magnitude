import { useCallback, useMemo } from "react"
import {
  Atom,
  Result,
  useAtomSet,
  useAtomValue,
} from "@effect-atom/atom-react"
import { Effect } from "effect"
import { type ModelId } from "@magnitudedev/sdk"
import type { HarnessId } from "../harness-connections/service"
import { OnboardingModelSetup } from "../local-models/setup"
import type { OnboardingModelRankingControls } from "../local-models/setup-state"
import { onboardingModelSetupViewAtom } from "../local-models/setup-view"
import { useAgentClient } from "../state/agent-client-context"

export const useOnboardingModelSetup = () => {
  const client = useAgentClient()
  const hardwareAtom = useMemo(() => Atom.make((get) =>
    get(client.Models.GetLocalEnvironment({})).result), [client])
  const view = useMemo(() => onboardingModelSetupViewAtom(client), [client])
  const retryAction = useMemo(() => client.runtime.fn(
    () => Effect.flatMap(OnboardingModelSetup, (setup) => setup.retry),
    { concurrent: true },
  ), [client])
  const openAction = useMemo(() => client.runtime.fn(
    () => Effect.flatMap(OnboardingModelSetup, (setup) => setup.open),
    { concurrent: true },
  ), [client])
  const selectAction = useMemo(() => client.runtime.fn<ModelId>()(
    (modelId) => Effect.flatMap(
      OnboardingModelSetup,
      (setup) => setup.select(modelId),
    ),
    { concurrent: true },
  ), [client])
  const setRankingControlsAction = useMemo(() => client.runtime.fn<OnboardingModelRankingControls>()(
    (controls) => Effect.flatMap(
      OnboardingModelSetup,
      (setup) => setup.setRankingControls(controls),
    ),
    { concurrent: true },
  ), [client])
  const cancelAction = useMemo(() => client.runtime.fn(
    () => Effect.flatMap(OnboardingModelSetup, (setup) => setup.cancel),
    { concurrent: true },
  ), [client])
  const chooseAnotherAction = useMemo(() => client.runtime.fn(
    () => Effect.flatMap(OnboardingModelSetup, (setup) => setup.chooseAnother),
    { concurrent: true },
  ), [client])
  const exitAction = useMemo(() => client.runtime.fn(
    () => Effect.flatMap(OnboardingModelSetup, (setup) => setup.exit),
    { concurrent: true },
  ), [client])
  const backAction = useMemo(() => client.runtime.fn(
    () => Effect.flatMap(OnboardingModelSetup, (setup) => setup.back),
    { concurrent: true },
  ), [client])
  const continueAction = useMemo(() => client.runtime.fn<{
    readonly harness: HarnessId
    readonly launchOnStartup: boolean
    readonly installSkill: boolean
  }>()(
    (input) => Effect.flatMap(
      OnboardingModelSetup,
      (setup) => setup.continueWithHarness(input.harness, input),
    ),
    { concurrent: true },
  ), [client])
  const retry = useAtomSet(retryAction)
  const open = useAtomSet(openAction)
  const select = useAtomSet(selectAction)
  const setRankingControls = useAtomSet(setRankingControlsAction)
  const cancel = useAtomSet(cancelAction)
  const chooseAnother = useAtomSet(chooseAnotherAction)
  const exit = useAtomSet(exitAction)
  const back = useAtomSet(backAction)
  const continueWithHarness = useAtomSet(continueAction)

  return {
    hardware: useAtomValue(hardwareAtom),
    view: useAtomValue(view),
    retry: useCallback(() => retry(), [retry]),
    open: useCallback(() => open(), [open]),
    select: useCallback((modelId: ModelId) => {
      select(modelId)
    }, [select]),
    setRankingControls: useCallback((controls: OnboardingModelRankingControls) => {
      setRankingControls(controls)
    }, [setRankingControls]),
    cancel: useCallback(() => cancel(), [cancel]),
    chooseAnother: useCallback(() => chooseAnother(), [chooseAnother]),
    back: useCallback(() => back(), [back]),
    continueWithHarness: useCallback((input: {
      readonly harness: HarnessId
      readonly launchOnStartup: boolean
      readonly installSkill: boolean
    }) => continueWithHarness(input), [continueWithHarness]),
    exit: useCallback(() => exit(), [exit]),
  }
}
