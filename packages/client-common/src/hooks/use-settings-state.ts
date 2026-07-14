/**
 * Shared provider-auth settings state for web, desktop, and CLI.
 * This settings path reads only masked summaries and invalidates model
 * discovery after auth mutations.
 */
import { useMemo } from "react"
import { useAtomValue, useAtomSet, Result } from "@effect-atom/atom-react"
import { useAgentClient } from "../state/agent-client-context"
import { apiKeyVerifiedAtom } from "../state/session-atoms"
import type { ProviderAuthSummary } from "@magnitudedev/sdk"

export interface ApiKeyState {
  readonly status: "none" | "loading" | "config"
  readonly maskedKey?: string
}

export interface UseSettingsStateResult {
  /** Masked auth state for every supported provider. */
  readonly providerAuths: readonly ProviderAuthSummary[] | null
  /** Current Magnitude API key state retained for login and usage screens. */
  readonly apiKey: ApiKeyState
  readonly keyAlreadySet: boolean
  readonly loading: boolean
  readonly failed: boolean
  readonly saveProviderApiKey: (providerId: string, key: string) => Promise<void>
  readonly disconnectProvider: (providerId: string) => Promise<void>
  readonly saveApiKey: (key: string) => Promise<void>
  readonly disconnectApiKey: () => Promise<void>
}

const MAGNITUDE_PROVIDER_ID = "magnitude"
const AUTH_REACTIVITY_KEYS = ["providerAuth", "modelConfig", "apiKey"] as const

export function useSettingsState(): UseSettingsStateResult {
  const client = useAgentClient()
  const setApiKeyVerified = useAtomSet(apiKeyVerifiedAtom)

  const result = useAtomValue(
    client.query("ListProviderAuthSummaries", {}, { reactivityKeys: ["providerAuth", "apiKey"] }),
  )

  const updateProviderAuth = useAtomSet(
    client.mutation("UpdateProviderAuth"),
    { mode: "promise" },
  )
  const removeProviderAuth = useAtomSet(
    client.mutation("RemoveProviderAuth"),
    { mode: "promise" },
  )

  const loading = Result.isInitial(result)
  const failed = Result.isFailure(result)
  const providerAuths = Result.match(result, {
    onInitial: () => null,
    onFailure: () => null,
    onSuccess: (success) => success.value.auths as readonly ProviderAuthSummary[],
  })
  const magnitudeAuth = providerAuths?.find((auth) => auth.providerId === MAGNITUDE_PROVIDER_ID)
  const keyAlreadySet = magnitudeAuth?.configured === true
  const apiKey: ApiKeyState = loading
    ? { status: "loading" }
    : keyAlreadySet
      ? { status: "config", ...(magnitudeAuth?.maskedKey ? { maskedKey: magnitudeAuth.maskedKey } : {}) }
      : { status: "none" }

  const saveProviderApiKey = useMemo(
    () => async (providerId: string, key: string): Promise<void> => {
      const trimmed = key.trim()
      if (!trimmed) throw new Error("API key is required")
      await updateProviderAuth({
        payload: { providerId, auth: { type: "api", key: trimmed } },
        reactivityKeys: [...AUTH_REACTIVITY_KEYS],
      })
    },
    [updateProviderAuth],
  )

  const disconnectProvider = useMemo(
    () => async (providerId: string): Promise<void> => {
      await removeProviderAuth({
        payload: { providerId },
        reactivityKeys: [...AUTH_REACTIVITY_KEYS],
      })
      if (providerId === MAGNITUDE_PROVIDER_ID) setApiKeyVerified(false)
    },
    [removeProviderAuth, setApiKeyVerified],
  )

  const saveApiKey = useMemo(
    () => (key: string) => saveProviderApiKey(MAGNITUDE_PROVIDER_ID, key),
    [saveProviderApiKey],
  )
  const disconnectApiKey = useMemo(
    () => () => disconnectProvider(MAGNITUDE_PROVIDER_ID),
    [disconnectProvider],
  )

  return {
    providerAuths,
    apiKey,
    keyAlreadySet,
    loading,
    failed,
    saveProviderApiKey,
    disconnectProvider,
    saveApiKey,
    disconnectApiKey,
  }
}
