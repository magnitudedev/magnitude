/**
 * Settings state hook — shared between web, desktop, and CLI.
 *
 * GetProviderAuth query + UpdateProviderAuth mutation.
 * Both apps use this identically.
 */
import { Option } from "effect"
import { useAtomValue, useAtomSet, Result } from "@effect-atom/atom-react"
import { useAgentClient } from "../state/agent-client-context"
import { apiKeyVerifiedAtom } from "../state/session-atoms"
import type { ProviderAuth } from "@magnitudedev/sdk"

export interface ApiKeyState {
  readonly status: "none" | "loading" | "config"
  readonly maskedKey?: string
}

export interface UseSettingsStateResult {
  /** Current API key state */
  apiKey: ApiKeyState
  /** Whether the key is already set (derived from query result) */
  keyAlreadySet: boolean
  /** Whether the query is loading */
  loading: boolean
  /** Save a new API key */
  saveApiKey: (key: string) => Promise<void>
  /** Disconnect (clear) the API key */
  disconnectApiKey: () => Promise<void>
}

function maskApiKey(key: string): string {
  const trimmed = key.trim()
  if (trimmed.length <= 12) return "•".repeat(Math.max(trimmed.length, 4))

  const lastUnderscore = trimmed.lastIndexOf("_")
  const head = lastUnderscore >= 0 && lastUnderscore < trimmed.length - 8
    ? trimmed.slice(0, lastUnderscore + 1 + 4)
    : trimmed.slice(0, 6)
  const tail = trimmed.slice(-4)
  return `${head}………${tail}`
}

const MAGNITUDE_PROVIDER_ID = "magnitude"

export function useSettingsState(): UseSettingsStateResult {
  const client = useAgentClient()
  const setApiKeyVerified = useAtomSet(apiKeyVerifiedAtom)

  const result = useAtomValue(
    client.query("GetProviderAuth", { providerId: MAGNITUDE_PROVIDER_ID }, { reactivityKeys: ["apiKey"] }),
  )

  const updateProviderAuth = useAtomSet(
    client.mutation("UpdateProviderAuth"),
    { mode: "promise" },
  )

  const loading = Result.isInitial(result)
  const keyAlreadySet = Result.match(result, {
    onInitial: () => false,
    onFailure: () => false,
    onSuccess: (s) => {
      const value = s.value as { auth: Option.Option<ProviderAuth> }
      return value.auth._tag === "Some" && value.auth.value.type === "api" && value.auth.value.key.trim().length > 0
    },
  })

  const apiKey: ApiKeyState = Result.match(result, {
    onInitial: () => ({ status: "loading" as const }),
    onFailure: () => ({ status: "none" as const }),
    onSuccess: (s) => {
      const value = s.value as { auth: Option.Option<ProviderAuth> }
      if (value.auth._tag === "Some" && value.auth.value.type === "api" && value.auth.value.key.trim().length > 0) {
        return { status: "config" as const, maskedKey: maskApiKey(value.auth.value.key) }
      }
      return { status: "none" as const }
    },
  })

  async function saveApiKey(key: string): Promise<void> {
    await updateProviderAuth({
      payload: { providerId: MAGNITUDE_PROVIDER_ID, auth: { type: "api", key } },
      reactivityKeys: ["apiKey"],
    })
  }

  async function disconnectApiKey(): Promise<void> {
    // Clear by setting an empty key — the server can handle this
    await updateProviderAuth({
      payload: { providerId: MAGNITUDE_PROVIDER_ID, auth: { type: "api", key: "" } },
      reactivityKeys: ["apiKey"],
    })
    setApiKeyVerified(false)
  }

  return { apiKey, keyAlreadySet, loading, saveApiKey, disconnectApiKey }
}
