import { useCallback, useMemo } from "react"
import { useAtomMount, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { useAgentClient } from "../state/agent-client-context"
import { onboardingAtoms } from "../onboarding/atoms"

export function useOnboardingState() {
  const client = useAgentClient()
  const atoms = useMemo(() => onboardingAtoms(client), [client])
  const updateMutation = useAtomSet(atoms.updateMutation)
  useAtomMount(atoms.mirrorInvalidationWatchAtom)
  useAtomMount(atoms.invalidationBridgeAtom)

  const update = useCallback((completed: boolean) => {
    updateMutation({ completed })
  }, [updateMutation])

  return {
    state: useAtomValue(atoms.resultAtom),
    updateResult: useAtomValue(atoms.updateMutation),
    update,
  }
}
