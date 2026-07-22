/**
 * LoginScreen — spec §10 "API key / login screen"
 *
 * Full-screen centered card. Shown when no API key is configured.
 * Checks GetProviderAuth — if a key is already set, the parent skips this screen.
 * Uses UpdateProviderAuth mutation to save the key.
 */
import { useState, useCallback, type ReactNode } from "react"
import { Loader2, ArrowRight } from "lucide-react"
import { Option } from "effect"
import { useAtomValue, useAtomSet, Result } from "@effect-atom/atom-react"
import { useAgentClient } from "@magnitudedev/client-common"
import { ProviderIdSchema } from "@magnitudedev/sdk"
import { apiKeyVerifiedAtom } from "../state/web-atoms"

const MAGNITUDE_PROVIDER_ID = ProviderIdSchema.make("magnitude")

export function LoginScreen(): ReactNode {
  const client = useAgentClient()
  const setVerified = useAtomSet(apiKeyVerifiedAtom)
  const [inputKey, setInputKey] = useState("")
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const updateProviderAuth = useAtomSet(client.mutation("UpdateProviderAuth"), { mode: "promise" })

  const handleConnect = useCallback(async () => {
    const trimmed = inputKey.trim()
    if (!trimmed) {
      setError("Please enter an API key.")
      return
    }
    setSubmitting(true)
    setError(null)
    try {
      await updateProviderAuth({ payload: { providerId: MAGNITUDE_PROVIDER_ID, auth: { type: "api", key: trimmed } } })
      setVerified(true)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to connect. Please check your key.")
    } finally {
      setSubmitting(false)
    }
  }, [inputKey, updateProviderAuth, setVerified])

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "Enter" && !submitting) {
        e.preventDefault()
        handleConnect()
      }
    },
    [handleConnect, submitting],
  )

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "var(--bg-base)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 200,
      }}
    >
      <div
        style={{
          background: "var(--bg-surface)",
          border: "1px solid var(--border-default)",
          borderRadius: 6,
          padding: 32,
          maxWidth: 420,
          width: "100%",
          display: "flex",
          flexDirection: "column",
          boxShadow: "0 16px 64px rgba(0,0,0,0.6)",
          animation: "fade-in 200ms ease-out",
        }}
      >
        {/* Title */}
        <h1
          style={{
            fontFamily: "var(--font-sans)",
            fontSize: 18,
            fontWeight: 600,
            color: "var(--fg-primary)",
            margin: 0,
            marginBottom: 4,
          }}
        >
          Welcome to Magnitude
        </h1>

        {/* Subtitle */}
        <p
          style={{
            fontFamily: "var(--font-sans)",
            fontSize: 14,
            color: "var(--fg-secondary)",
            margin: 0,
            marginBottom: 24,
          }}
        >
          Enter your Magnitude API key to connect.
        </p>

        {/* Input */}
        <input
          type="password"
          value={inputKey}
          onChange={(e) => setInputKey(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="mg_..."
          autoFocus
          disabled={submitting}
          className="hover-focus-border"
          data-error={error ? "true" : undefined}
          style={{
            width: "100%",
            background: "var(--bg-input)",
            border: `1px solid ${error ? "var(--accent-error)" : "var(--border-default)"}`,
            borderRadius: 4,
            padding: "8px 12px",
            fontFamily: "var(--font-mono)",
            fontSize: 14,
            color: "var(--fg-primary)",
            outline: "none",
            boxSizing: "border-box",
            marginBottom: 16,
            transition: "border 100ms",
          }}
        />

        {/* Error */}
        {error && (
          <p
            style={{
              fontFamily: "var(--font-sans)",
              fontSize: 13,
              color: "var(--accent-error)",
              margin: 0,
              marginBottom: 16,
            }}
          >
            {error}
          </p>
        )}

        {/* Connect button */}
        <button
          onClick={handleConnect}
          disabled={submitting || !inputKey.trim()}
          className="hover-opacity"
          data-disabled={submitting || !inputKey.trim() ? "true" : "false"}
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 6,
            background: "var(--accent-primary)",
            color: "white",
            border: "none",
            borderRadius: 4,
            padding: "8px 16px",
            fontFamily: "var(--font-sans)",
            fontSize: 14,
            fontWeight: 500,
            cursor: submitting || !inputKey.trim() ? "default" : "pointer",
            opacity: submitting || !inputKey.trim() ? 0.5 : 1,
            transition: "opacity 100ms",
          }}
        >
          {submitting ? (
            <>
              <Loader2 size={16} style={{ animation: "spin 1s linear infinite" }} />
              <span>Connecting...</span>
            </>
          ) : (
            <>
              <span>Connect</span>
              <ArrowRight size={16} />
            </>
          )}
        </button>
      </div>
    </div>
  )
}
