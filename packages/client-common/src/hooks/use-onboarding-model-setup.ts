import { useCallback, useMemo } from "react"
import {
  Atom,
  Result,
  useAtomSet,
  useAtomValue,
} from "@effect-atom/atom-react"
import { Effect } from "effect"
import { type ProviderModelId } from "@magnitudedev/sdk"
import { OnboardingModelSetup } from "../local-models/setup"
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
  const selectAction = useMemo(() => client.runtime.fn<ProviderModelId>()(
    (modelId) => Effect.flatMap(
      OnboardingModelSetup,
      (setup) => setup.select(modelId),
    ),
    { concurrent: true },
  ), [client])
  const cancelAction = useMemo(() => client.runtime.fn(
    () => Effect.flatMap(OnboardingModelSetup, (setup) => setup.cancel),
    { concurrent: true },
  ), [client])
  const exitAction = useMemo(() => client.runtime.fn(
    () => Effect.flatMap(OnboardingModelSetup, (setup) => setup.exit),
    { concurrent: true },
  ), [client])
  const retry = useAtomSet(retryAction)
  const open = useAtomSet(openAction)
  const select = useAtomSet(selectAction)
  const cancel = useAtomSet(cancelAction)
  const exit = useAtomSet(exitAction)

  return {
    hardware: useAtomValue(hardwareAtom),
    view: useAtomValue(view),
    retry: useCallback(() => retry(), [retry]),
    open: useCallback(() => open(), [open]),
    select: useCallback((modelId: ProviderModelId) => {
      select(modelId)
    }, [select]),
    cancel: useCallback(() => cancel(), [cancel]),
    exit: useCallback(() => exit(), [exit]),
  }
}
