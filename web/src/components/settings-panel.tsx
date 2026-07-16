/**
 * SettingsPanel — unified settings + usage panel
 *
 * Replaces the entire chat-column when open. Has a tab bar to switch
 * between Settings (API key + roles) and Usage (balance + charts).
 * No modal chrome — fills its parent container.
 */
import { useState, useCallback, type ReactNode } from "react"
import { Option } from "effect"
import { Result } from "@effect-atom/atom-react"
import { formatTokensCompact } from "@magnitudedev/client-common"
import { AlertTriangle } from "lucide-react"
import type { BalanceResponse, UsagePeriod, SlotId, ReasoningEffort } from "@magnitudedev/sdk"
import { ModelCatalogLifecycle, ModelSlotsLifecycle, SLOT_DISPLAY_NAMES, SLOT_DESCRIPTIONS, SLOT_IDS, DEFAULT_REASONING_EFFORT } from "@magnitudedev/sdk"
import type { UseModelConfigResult } from "@magnitudedev/client-common"

export type { UsagePeriod } from "@magnitudedev/sdk"

type Tab = "settings" | "usage"

const catalogStateOf = (modelConfig: UseModelConfigResult) =>
  Option.map(Result.value(modelConfig.catalog), ({ state }) => state)

const catalogModelsOf = (modelConfig: UseModelConfigResult) => Option.flatMap(
  catalogStateOf(modelConfig),
  (state) => ModelCatalogLifecycle.match(state, {
    loading: () => Option.none(),
    ready: ({ models }) => Option.some(models),
    refreshing: ({ models }) => Option.some(models),
    degraded: ({ models }) => Option.some(models),
    unavailable: () => Option.none(),
  }),
)

const slotConfigurationOf = (modelConfig: UseModelConfigResult) => Option.flatMap(
  Result.value(modelConfig.slots),
  ({ state }) => ModelSlotsLifecycle.match(state, {
    loading: () => Option.none(),
    ready: ({ config }) => Option.some(config),
    refreshing: ({ config }) => Option.some(config),
    degraded: ({ config }) => Option.some(config),
    unavailable: ({ config }) => Option.some(config),
  }),
)

// ── Settings types ──

export interface SlotEntry {
  slotId: SlotId
  label: string
  description: string
  modelDisplayName: string
  contextWindow: number | null
}

export interface ApiKeyState {
  /** "config" = connected via config, "none" = not connected */
  status: "config" | "none"
  /** Masked key for display */
  maskedKey?: string
  /** Error message */
  error?: string
}

// ── Usage types ──

const PERIODS: UsagePeriod[] = ["24h", "3d", "7d", "14d", "30d", "all"]

// ── Panel props ──

export interface SettingsPanelProps {
  /** API key state */
  apiKey: ApiKeyState
  /** Save a new API key */
  onSaveApiKey?: (key: string) => Promise<void>
  /** Disconnect API key */
  onDisconnectApiKey?: () => Promise<void>
  /** Slot entries for display */
  slots?: SlotEntry[]
  /** Model config hook result for model selection UI */
  modelConfig?: UseModelConfigResult
  /** Usage loading state */
  usageLoading?: boolean
  /** Usage error message */
  usageError?: string | null
  /** GetBalance response */
  usageData?: BalanceResponse | null
  /** Current usage period */
  usagePeriod: UsagePeriod
  /** Called when usage period changes */
  onUsagePeriodChange: (period: UsagePeriod) => void
  /** Which tab to show first */
  initialTab?: Tab
}

export function SettingsPanel({
  apiKey,
  onSaveApiKey,
  onDisconnectApiKey,
  slots = [],
  modelConfig,
  usageLoading = false,
  usageError = null,
  usageData = null,
  usagePeriod,
  onUsagePeriodChange,
  initialTab = "settings",
}: SettingsPanelProps): ReactNode {
  const [tab, setTab] = useState<Tab>(initialTab)

  return (
    <div style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
      {/* Tab bar */}
      <div
        style={{
          display: "flex",
          gap: 0,
          borderBottom: "1px solid var(--border-subtle)",
          flexShrink: 0,
        }}
      >
        {(["settings", "usage"] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              background: "transparent",
              border: "none",
              borderBottom: t === tab ? "2px solid var(--accent-primary)" : "2px solid transparent",
              color: t === tab ? "var(--accent-primary)" : "var(--fg-secondary)",
              fontFamily: "var(--font-sans)",
              fontSize: 13,
              fontWeight: 500,
              padding: "10px 16px",
              cursor: "pointer",
              marginBottom: -1,
              textTransform: "capitalize",
            }}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div style={{ overflowY: "auto", padding: 16, flex: 1 }}>
        {tab === "settings" ? (
          <SettingsTab
            apiKey={apiKey}
            onSaveApiKey={onSaveApiKey}
            onDisconnectApiKey={onDisconnectApiKey}
            slots={slots}
            modelConfig={modelConfig}
          />
        ) : (
          <UsageTab
            loading={usageLoading}
            error={usageError}
            data={usageData}
            period={usagePeriod}
            onPeriodChange={onUsagePeriodChange}
          />
        )}
      </div>
    </div>
  )
}

// ── Settings tab ──

function SettingsTab({
  apiKey,
  onSaveApiKey,
  onDisconnectApiKey,
  slots,
  modelConfig,
}: {
  apiKey: ApiKeyState
  onSaveApiKey?: (key: string) => Promise<void>
  onDisconnectApiKey?: () => Promise<void>
  slots: SlotEntry[]
  modelConfig?: UseModelConfigResult
}): ReactNode {
  const [mode, setMode] = useState<"view" | "edit" | "disconnect-confirm">("view")
  const [inputKey, setInputKey] = useState("")
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const catalogState = modelConfig ? catalogStateOf(modelConfig) : Option.none()
  const catalogLoading = Option.match(catalogState, {
    onNone: () => modelConfig !== undefined && !Result.isFailure(modelConfig.catalog),
    onSome: (state) => ModelCatalogLifecycle.is(state, "loading"),
  })
  const catalogRefreshing = modelConfig !== undefined && (
    Result.isWaiting(modelConfig.catalogRefresh)
    || Option.exists(catalogState, (state) => ModelCatalogLifecycle.is(state, "refreshing"))
  )

  const handleSave = useCallback(async () => {
    if (!onSaveApiKey) return
    setSaving(true)
    setError(null)
    try {
      await onSaveApiKey(inputKey)
      setMode("view")
      setInputKey("")
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save API key")
    } finally {
      setSaving(false)
    }
  }, [onSaveApiKey, inputKey])

  const handleDisconnect = useCallback(async () => {
    if (!onDisconnectApiKey) return
    setSaving(true)
    try {
      await onDisconnectApiKey()
      setMode("view")
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to disconnect")
    } finally {
      setSaving(false)
    }
  }, [onDisconnectApiKey])

  return (
    <div className="settings-api-key-section">
      <h3
        style={{
          fontFamily: "var(--font-sans)",
          fontSize: 15,
          fontWeight: 600,
          color: "var(--fg-primary)",
          marginBottom: 12,
        }}
      >
        Magnitude
      </h3>

      {mode === "view" && apiKey.status === "config" && (
        <>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
            <StatusDot color="var(--accent-success)" />
            <span style={{ fontSize: 13, color: "var(--fg-primary)" }}>
              Connected {apiKey.maskedKey && `(${apiKey.maskedKey})`}
            </span>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <SettingsButton onClick={() => setMode("edit")}>Update key</SettingsButton>
            <SettingsButton onClick={() => setMode("disconnect-confirm")} danger>
              Disconnect
            </SettingsButton>
          </div>
        </>
      )}

      {mode === "view" && apiKey.status === "none" && (
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <StatusDot color="var(--fg-tertiary)" />
          <SettingsButton onClick={() => setMode("edit")}>Set API key</SettingsButton>
        </div>
      )}

      {mode === "edit" && (
        <div>
          <input
            type="password"
            value={inputKey}
            onChange={(e) => setInputKey(e.target.value)}
            placeholder="Enter API key..."
            autoFocus
            style={{
              width: "100%",
              maxWidth: 400,
              padding: "8px 10px",
              background: "var(--bg-input)",
              border: `1px solid ${error ? "var(--accent-error)" : "var(--accent-primary)"}`,
              borderRadius: 4,
              color: "var(--fg-primary)",
              fontFamily: "var(--font-mono)",
              fontSize: 13,
              outline: "none",
              marginBottom: 8,
            }}
          />
          <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
            <SettingsButton onClick={handleSave} disabled={saving || !inputKey.trim()}>
              {saving ? "Saving..." : "Save"}
            </SettingsButton>
            <SettingsButton onClick={() => { setMode("view"); setError(null); setInputKey("") }}>
              Cancel
            </SettingsButton>
          </div>
          {error && (
            <p style={{ fontSize: 13, color: "var(--accent-error)" }}>{error}</p>
          )}
        </div>
      )}

      {mode === "disconnect-confirm" && (
        <div>
          <p style={{ fontSize: 13, color: "var(--fg-secondary)", marginBottom: 8 }}>
            Disconnect this key? You will need to set another to reconnect.
          </p>
          <div style={{ display: "flex", gap: 8 }}>
            <SettingsButton onClick={handleDisconnect} disabled={saving} danger>
              {saving ? "Disconnecting..." : "Yes, disconnect"}
            </SettingsButton>
            <SettingsButton onClick={() => setMode("view")}>Cancel</SettingsButton>
          </div>
        </div>
      )}

      {/* Model Selection Section */}
      {slots.length > 0 && (
        <div style={{ marginTop: 24 }}>
          <div style={{ height: 1, background: "var(--border-subtle)", marginBottom: 16 }} />
          <h3
            style={{
              fontFamily: "var(--font-sans)",
              fontSize: 15,
              fontWeight: 600,
              color: "var(--fg-primary)",
              marginBottom: 12,
            }}
          >
            Model Selection
          </h3>
          {modelConfig && Result.isFailure(modelConfig.catalogRefresh) && (
            <div style={{ marginBottom: 8, fontSize: 12, color: "var(--accent-error)" }}>
              Failed to request a model catalog refresh.
            </div>
          )}
          {modelConfig && Result.isFailure(modelConfig.slotUpdate) && (
            <div style={{ marginBottom: 8, fontSize: 12, color: "var(--accent-error)" }}>
              Failed to update model configuration.
            </div>
          )}
          {slots.map((entry, i) => (
            <SlotCard
              key={entry.slotId}
              entry={entry}
              modelConfig={modelConfig}
              isLast={i === slots.length - 1}
            />
          ))}
          {modelConfig && (
            <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
              <SettingsButton
                onClick={() => { void modelConfig.refreshModels() }}
                disabled={catalogLoading || catalogRefreshing}
              >
                {catalogRefreshing ? "Refreshing..." : "Refresh models"}
              </SettingsButton>
              <SettingsButton
                onClick={() => { void modelConfig.resetToDefaults() }}
                disabled={catalogLoading || catalogRefreshing}
              >
                Reset to defaults
              </SettingsButton>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── Usage tab ──

function UsageTab({
  loading,
  error,
  data,
  period,
  onPeriodChange,
}: {
  loading: boolean
  error: string | null
  data: BalanceResponse | null
  period: UsagePeriod
  onPeriodChange: (period: UsagePeriod) => void
}): ReactNode {
  const balance = data?.data.balance.cents != null ? data.data.balance.cents / 100 : null
  const autoReload = data?.data.autoReload ?? null
  const autoReloadError = autoReload?.lastFailure ? autoReload.lastFailure.reason : null
  const hasPaymentMethod = data?.data.hasPaymentMethod ?? false
  const totals = data?.data.usage.totals ?? null
  const byModel = data?.data.usage.byModel ?? []
  const dailyTokens = data?.data.usage.dailyTokens ?? []
  const totalCostCents = byModel.reduce((sum, m) => sum + m.costCents, 0)

  return (
    <>
      {loading && (
        <div style={{ color: "var(--fg-tertiary)", fontSize: 13, fontFamily: "var(--font-mono)" }}>
          Loading<span className="animate-blink">...</span>
        </div>
      )}

      {error && !loading && (
        <div style={{ color: "var(--accent-error)", fontSize: 13, fontFamily: "var(--font-mono)" }}>
          Failed to load usage: {error}
        </div>
      )}

      {!loading && !error && data && (
        <>
          {/* Balance strip */}
          <div style={{ marginBottom: 16 }}>
            {balance != null && (
              <>
                <span style={{ fontSize: 14, fontWeight: 600, color: "var(--fg-primary)" }}>
                  Balance:{" "}
                </span>
                <span
                  style={{
                    fontSize: 14,
                    fontWeight: 600,
                    color:
                      balance <= 1
                        ? "var(--accent-error)"
                        : "var(--accent-success)",
                  }}
                >
                  ${balance.toFixed(2)}
                </span>
              </>
            )}
            <div style={{ marginTop: 4 }}>
              <span style={{ fontSize: 12, color: "var(--fg-secondary)" }}>
                Auto-reload:{" "}
                {autoReload && autoReload.enabled
                  ? `at $${(autoReload.thresholdCents / 100).toFixed(2)} \u2192 +$${(autoReload.amountCents / 100).toFixed(2)}`
                  : "off"}
              </span>
            </div>
            <div style={{ marginTop: 4, display: "flex", alignItems: "center", gap: 6 }}>
              <span
                style={{
                  display: "inline-block",
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: hasPaymentMethod
                    ? "var(--accent-success)"
                    : "var(--fg-tertiary)",
                }}
              />
              <span style={{ fontSize: 12, color: "var(--fg-secondary)" }}>
                {hasPaymentMethod
                  ? "Payment method on file"
                  : "No payment method"}
              </span>
            </div>

            {autoReloadError && (
              <div
                style={{
                  marginTop: 8,
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  padding: "6px 10px",
                  background: "var(--tint-warning)",
                  borderRadius: 4,
                  borderLeft: "3px solid var(--accent-warning)",
                }}
              >
                <AlertTriangle size={14} style={{ color: "var(--accent-warning)" }} />
                <span style={{ fontSize: 12, color: "var(--fg-secondary)" }}>
                  {autoReloadError}
                </span>
              </div>
            )}
          </div>

          {/* Period tabs */}
          <div
            role="tablist"
            aria-label="Usage period"
            style={{
              display: "flex",
              gap: 12,
              borderBottom: "1px solid var(--border-subtle)",
              marginBottom: 12,
              paddingBottom: 0,
            }}
          >
            {PERIODS.map((p) => (
              <button
                key={p}
                role="tab"
                aria-selected={p === period}
                onClick={() => onPeriodChange(p)}
                style={{
                  background: "transparent",
                  border: "none",
                  borderBottom: p === period ? "2px solid var(--accent-primary)" : "2px solid transparent",
                  color: p === period ? "var(--accent-primary)" : "var(--fg-secondary)",
                  fontFamily: "var(--font-sans)",
                  fontSize: 13,
                  padding: "4px 0",
                  cursor: "pointer",
                  marginBottom: -1,
                }}
              >
                {p}
              </button>
            ))}
          </div>

          {/* Period summary */}
          {totals && (
            <div style={{ marginBottom: 16, fontSize: 13, color: "var(--fg-secondary)" }}>
              <span>{totals.requestCount} reqs</span>
              {" \u00B7 "}
              <span>${(totals.costCents / 100).toFixed(2)} spend</span>
              {" \u00B7 "}
              <span>
                {formatTokensCompact(totals.inputTokens)} in /{" "}
                {formatTokensCompact(totals.outputTokens)} out
              </span>
            </div>
          )}

          {/* Top models */}
          {byModel.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <h4
                style={{
                  fontFamily: "var(--font-sans)",
                  fontSize: 13,
                  fontWeight: 600,
                  color: "var(--fg-primary)",
                  marginBottom: 8,
                }}
              >
                Top Models
              </h4>
              {byModel.map((m) => {
                const percentage = totalCostCents > 0 ? (m.costCents / totalCostCents) * 100 : 0
                return (
                  <div
                    key={m.model}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 8,
                      marginBottom: 4,
                      fontSize: 12,
                    }}
                  >
                    <span
                      style={{
                        color: "var(--fg-primary)",
                        flex: "0 0 140px",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {m.model}
                    </span>
                    <div
                      style={{
                        width: 100,
                        height: 6,
                        background: "var(--border-default)",
                        borderRadius: 3,
                        overflow: "hidden",
                        flexShrink: 0,
                      }}
                    >
                      <div
                        style={{
                          width: `${Math.min(100, percentage)}%`,
                          height: "100%",
                          background: "var(--accent-primary)",
                          borderRadius: 3,
                        }}
                      />
                    </div>
                    <span style={{ color: "var(--fg-secondary)" }}>{m.requestCount} reqs</span>
                    <span style={{ color: "var(--fg-secondary)" }}>${(m.costCents / 100).toFixed(2)}</span>
                  </div>
                )
              })}
            </div>
          )}

          {/* Daily tokens chart */}
          {dailyTokens.length > 0 && (
            <div>
              <h4
                style={{
                  fontFamily: "var(--font-sans)",
                  fontSize: 13,
                  fontWeight: 600,
                  color: "var(--fg-primary)",
                  marginBottom: 8,
                }}
              >
                Daily Tokens
              </h4>
              {dailyTokens.map((d) => {
                const total = d.inputTokens + d.outputTokens
                const maxTotal = Math.max(
                  ...dailyTokens.map((x) => x.inputTokens + x.outputTokens),
                  1,
                )
                const barWidth = (total / maxTotal) * 120
                return (
                  <div
                    key={d.date}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 8,
                      marginBottom: 4,
                      fontSize: 12,
                    }}
                  >
                    <span style={{ color: "var(--fg-secondary)", flex: "0 0 60px" }}>{d.date}</span>
                    <div
                      style={{
                        width: 120,
                        flexShrink: 0,
                      }}
                    >
                      <div
                        style={{
                          width: `${barWidth}px`,
                          height: 6,
                          background: "var(--accent-primary)",
                          borderRadius: 3,
                        }}
                      />
                    </div>
                    <span style={{ color: "var(--fg-secondary)" }}>
                      {formatTokensCompact(d.inputTokens)} in /{" "}
                      {formatTokensCompact(d.outputTokens)} out
                    </span>
                    <span
                      style={{
                        color: "var(--fg-tertiary)",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {d.topModel ?? ""}
                    </span>
                  </div>
                )
              })}
            </div>
          )}
        </>
      )}

      {!loading && !error && !data && (
        <div style={{ padding: "24px 0", textAlign: "center", color: "var(--fg-tertiary)", fontSize: 13, fontFamily: "var(--font-sans)" }}>
          Usage data is not available.
          <br />
          <span style={{ fontSize: 12, color: "var(--fg-quaternary)" }}>
            Connect a Magnitude account to view usage statistics.
          </span>
        </div>
      )}
    </>
  )
}

// ── Helpers ──

const REASONING_OPTIONS: { value: ReasoningEffort; label: string }[] = [
  { value: "none", label: "None" },
  { value: "low", label: "Low" },
  { value: "medium", label: "Medium" },
  { value: "high", label: "High" },
  { value: "max", label: "Max" },
]

function formatContextWindowCompact(tokens: number): string {
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(1)}M`
  if (tokens >= 1_000) return `${Math.round(tokens / 1000)}K`
  return `${tokens}`
}

function formatPricing(pricing: { input: number; output: number }): string {
  return `$${pricing.input.toFixed(2)}/$${pricing.output.toFixed(2)}`
}

function SlotCard({
  entry,
  modelConfig,
  isLast,
}: {
  entry: SlotEntry
  modelConfig?: UseModelConfigResult
  isLast: boolean
}): ReactNode {
  const slotId = entry.slotId
  const catalogState = modelConfig ? catalogStateOf(modelConfig) : Option.none()
  const models = modelConfig ? Option.getOrNull(catalogModelsOf(modelConfig)) : null
  const slotConfiguration = modelConfig ? Option.getOrNull(slotConfigurationOf(modelConfig)) : null
  const currentOverride = slotConfiguration?.slots[slotId] ?? null
  const defaultEffort = DEFAULT_REASONING_EFFORT[slotId]
  const currentEffort = currentOverride?.reasoningEffort ?? defaultEffort
  const transportFailed = modelConfig !== undefined && Result.isFailure(modelConfig.catalog)
  const transportHasSnapshot = modelConfig !== undefined && Option.isSome(Result.value(modelConfig.catalog))
  const loading = Option.match(catalogState, {
    onNone: () => modelConfig !== undefined && !transportFailed,
    onSome: (state) => ModelCatalogLifecycle.is(state, "loading"),
  })

  // Find the currently effective model: user override, then first model for this slot, then first overall
  const effectiveModelId = currentOverride?.providerModelId
    ?? models?.find(m => m.slots?.includes(slotId))?.providerModelId
    ?? models?.[0]?.providerModelId
    ?? null

  const handleModelChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const value = e.target.value
      if (modelConfig) {
        if (!value) {
          void modelConfig.updateSlotModel(slotId, null, null)
        } else {
          const model = models?.find(m => m.providerModelId === value)
          if (model) {
            void modelConfig.updateSlotModel(slotId, model.providerId, model.providerModelId)
          }
        }
      }
    },
    [modelConfig, slotId, models],
  )

  const handleEffortChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const value = e.target.value as ReasoningEffort | "default"
      if (modelConfig) {
        void modelConfig.updateSlotReasoning(slotId, value === "default" ? null : value)
      }
    },
    [modelConfig, slotId],
  )

  return (
    <div style={{ marginBottom: isLast ? 0 : 16 }}>
      <div style={{ marginBottom: 6 }}>
        <span
          style={{
            fontFamily: "var(--font-sans)",
            fontSize: 13,
            fontWeight: 600,
            color: "var(--accent-primary)",
          }}
        >
          {entry.label} Model
        </span>
      </div>

      {(transportFailed && !transportHasSnapshot)
        || Option.exists(catalogState, (state) => ModelCatalogLifecycle.is(state, "unavailable")) ? (
        <div style={{ marginBottom: 4 }}>
          <span style={{ fontSize: 13, color: "var(--accent-error)" }}>
            Unable to load available models
          </span>
        </div>
      ) : loading && models === null ? (
        <div style={{ marginBottom: 4 }}>
          <div style={{
            height: 32,
            width: 220,
            background: "var(--bg-surface)",
            borderRadius: 4,
            opacity: 0.5,
          }} />
        </div>
      ) : (
        <div>
          {transportFailed && transportHasSnapshot && (
            <div style={{ marginBottom: 6, fontSize: 12, color: "var(--accent-error)" }}>
              Lost contact with the model catalog; showing the last received state.
            </div>
          )}
          {Option.exists(catalogState, (state) =>
            ModelCatalogLifecycle.is(state, "degraded")
            || (ModelCatalogLifecycle.is(state, "refreshing") && state.failures.length > 0)) && (
            <div style={{ marginBottom: 6, fontSize: 12, color: "var(--accent-error)" }}>
              Some model providers are unavailable; showing available or last known models.
            </div>
          )}
          <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 4 }}>
          {/* Model dropdown */}
          <select
            value={effectiveModelId ?? ""}
            onChange={handleModelChange}
            style={{
              padding: "6px 10px",
              background: "var(--bg-input)",
              border: "1px solid var(--border-default)",
              borderRadius: 4,
              color: "var(--fg-primary)",
              fontFamily: "var(--font-sans)",
              fontSize: 13,
              outline: "none",
              cursor: "pointer",
              minWidth: 200,
            }}
          >
            {models && models.length > 0 ? (
              models.map((model) => (
                <option key={model.providerModelId} value={model.providerModelId}>
                  {model.displayName} — {formatContextWindowCompact(model.contextWindow)} ctx
                  {model.pricing ? ` — ${formatPricing(model.pricing)}` : ""}
                </option>
              ))
            ) : (
              <option value="">{entry.modelDisplayName}</option>
            )}
          </select>

          {/* Thinking level dropdown */}
          <select
            value={currentOverride?.reasoningEffort ?? "default"}
            onChange={handleEffortChange}
            style={{
              padding: "6px 10px",
              background: "var(--bg-input)",
              border: "1px solid var(--border-default)",
              borderRadius: 4,
              color: "var(--fg-primary)",
              fontFamily: "var(--font-sans)",
              fontSize: 13,
              outline: "none",
              cursor: "pointer",
            }}
          >
            <option value="default">
              {REASONING_OPTIONS.find((opt) => opt.value === defaultEffort)?.label} (default)
            </option>
            {REASONING_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
          </div>
        </div>
      )}

      <p style={{ fontSize: 12, color: "var(--fg-secondary)" }}>{entry.description}</p>
    </div>
  )
}

function StatusDot({ color }: { color: string }): ReactNode {
  return (
    <span
      style={{
        display: "inline-block",
        width: 8,
        height: 8,
        borderRadius: "50%",
        background: color,
        flexShrink: 0,
      }}
    />
  )
}

function SettingsButton({
  children,
  onClick,
  danger,
  disabled,
}: {
  children: ReactNode
  onClick: () => void
  danger?: boolean
  disabled?: boolean
}): ReactNode {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={danger ? "hover-danger-button" : "hover-surface"}
      style={{
        padding: "6px 12px",
        background: "transparent",
        border: `1px solid ${danger ? "var(--accent-error)" : "var(--border-default)"}`,
        borderRadius: 4,
        color: danger ? "var(--accent-error)" : "var(--fg-primary)",
        fontFamily: "var(--font-sans)",
        fontSize: 13,
        cursor: disabled ? "default" : "pointer",
        opacity: disabled ? 0.5 : 1,
        transition: "background 100ms",
      }}
    >
      {children}
    </button>
  )
}
