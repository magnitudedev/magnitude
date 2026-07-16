/**
 * Settings state hook — shared between web, desktop, and CLI.
 *
 * GetProviderAuth query + UpdateProviderAuth mutation.
 * Both apps use this identically.
 */
import { useMemo } from "react"
import { useAtomValue, useAtomSet, Result } from "@effect-atom/atom-react"
import { Option } from "effect"
import { ProviderIdSchema } from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"

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
  /** Authoritative provider-auth query failure */
  loadError: string | null
  /** Whether provider auth is being changed */
  saving: boolean
  /** Authoritative provider-auth mutation failure */
  saveError: string | null
  /** Save a new API key */
  saveApiKey: (key: string) => void
  /** Disconnect (clear) the API key */
  disconnectApiKey: () => void
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

const MAGNITUDE_PROVIDER_ID = ProviderIdSchema.make("magnitude")

export function useSettingsState(): UseSettingsStateResult {
  const client = useAgentClient()

  const queryAtom = useMemo(
    () => client.query("GetProviderAuth", { providerId: MAGNITUDE_PROVIDER_ID }, { reactivityKeys: ["apiKey"] }),
    [client],
  )
  const result = useAtomValue(queryAtom)

  const updateProviderAuthAtom = useMemo(() => client.mutation("UpdateProviderAuth"), [client])
  const updateProviderAuthResult = useAtomValue(updateProviderAuthAtom)
  const updateProviderAuth = useAtomSet(updateProviderAuthAtom)

  const saving = Result.isWaiting(updateProviderAuthResult)
  const saveError = Result.isFailure(updateProviderAuthResult)
    ? "Failed to update the Magnitude API key"
    : null

  const snapshot = Result.value(result)
  const loading = Result.isWaiting(result) && Option.isNone(snapshot)
  const loadError = Result.isFailure(result)
    ? "Failed to read the Magnitude API key configuration"
    : null
  const configuredKey = Option.flatMap(snapshot, ({ auth }) => Option.flatMap(auth, (value) =>
    value.type === "api" && value.key.trim().length > 0 ? Option.some(value) : Option.none()))
  const keyAlreadySet = Option.isSome(configuredKey)

  const apiKey: ApiKeyState = Option.match(snapshot, {
    onNone: () => Result.isFailure(result) ? { status: "none" } : { status: "loading" },
    onSome: () => Option.match(configuredKey, {
      onNone: () => ({ status: "none" }),
      onSome: (auth) => ({ status: "config", maskedKey: maskApiKey(auth.key) }),
    }),
  })

  function saveApiKey(key: string): void {
    updateProviderAuth({
      payload: { providerId: MAGNITUDE_PROVIDER_ID, auth: { type: "api", key } },
      reactivityKeys: ["apiKey", "modelCatalog", "modelSlots"],
    })
  }

  function disconnectApiKey(): void {
    // Clear by setting an empty key — the server can handle this
    updateProviderAuth({
      payload: { providerId: MAGNITUDE_PROVIDER_ID, auth: { type: "api", key: "" } },
      reactivityKeys: ["apiKey", "modelCatalog", "modelSlots"],
    })
  }

  return {
    apiKey,
    keyAlreadySet,
    loading,
    loadError,
    saving,
    saveError,
    saveApiKey,
    disconnectApiKey,
  }
}
