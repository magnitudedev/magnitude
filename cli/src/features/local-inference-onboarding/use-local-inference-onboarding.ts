import { useAtomValue } from "@effect-atom/atom-react"
import {
  useLocalInferenceState,
  useSettingsState,
} from "@magnitudedev/client-common"
import { authSourceAtom } from "../../state/cli-atoms"

/**
 * Pure CLI composition of shared local-inference and provider-auth domains.
 * Server facts and operation state remain owned by the client-common AtomRpc
 * hooks; this layer only adds the CLI-specific environment-auth signal.
 */
export function useLocalInferenceOnboarding() {
  const localInference = useLocalInferenceState()
  const settings = useSettingsState()
  const authSource = useAtomValue(authSourceAtom)

  return {
    ...localInference,
    configureCloud: settings.saveApiKey,
    cloudKeyAlreadySet: settings.keyAlreadySet || authSource.source === "env",
    busy: localInference.busy || settings.saving,
    error: localInference.error ?? settings.saveError,
  }
}
