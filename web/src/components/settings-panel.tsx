/**
 * SettingsPanel — unified settings + usage panel
 *
 * Replaces the entire chat-column when open. Has a tab bar to switch
 * between Settings (API key + roles) and Usage (subscription limits + charts).
 * No modal chrome — fills its parent container.
 */
import { useState, useCallback, type ReactNode } from "react"
import { Option } from "effect"
import { Result } from "@effect-atom/atom-react"
import { formatTokensCompact, reasoningEffortControl, reasoningPropertyLabel, selectedSlotModel, useLocalInferenceState, visionPropertyLabel } from "@magnitudedev/client-common"
import { AlertTriangle } from "lucide-react"
import type { CloudUsageResponse, UsagePeriod, SlotId, ReasoningEffort, LocalModelChoice } from "@magnitudedev/sdk"
import { ModelCatalogLifecycle } from "@magnitudedev/sdk"
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
  /** GetCloudUsage response */
  usageData?: CloudUsageResponse | null
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
  const localInference = useLocalInferenceState()
  const localSnapshot = Result.value(localInference.state)
  const localChoices = Option.match(localSnapshot, {
    onNone: () => [] as const,
    onSome: ({ choices }) => choices,
  })
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
              localChoices={localChoices}
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
  data: CloudUsageResponse | null
  period: UsagePeriod
  onPeriodChange: (period: UsagePeriod) => void
}): ReactNode {
  const subscription = data?.data.subscription ?? null
  const usageWindows = data?.data.usageWindows ?? {}
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
          {/* Cloud subscription and current limit windows */}
          <div style={{ marginBottom: 16 }}>
            {subscription && (
              <div style={{ fontSize: 14, fontWeight: 600, color: "var(--fg-primary)" }}>
                Cloud subscription: <span style={{ color: "var(--accent-primary)" }}>
                  {subscription.status === "active" ? subscription.plan.label : "Not subscribed"}
                </span>
                <span style={{ color: "var(--fg-secondary)", fontWeight: 400 }}>
                  {subscription.status === "active"
                    ? " · $20/month"
                    : " · $10 first month, then $20/month"}
                </span>
              </div>
            )}
            {subscription?.status === "not_subscribed" && (
              <div style={{ marginTop: 4, fontSize: 12, color: "var(--fg-secondary)" }}>
                Magnitude Pro is required to use cloud models.
              </div>
            )}
            {Object.entries(usageWindows).map(([window, budget]) => budget && (
              <div key={window} style={{ marginTop: 4, fontSize: 12, color: "var(--fg-secondary)" }}>
                {window === "five_hour" ? "5h" : window}: ${(budget.usedCents / 100).toFixed(2)} of ${(budget.limitCents / 100).toFixed(2)}
              </div>
            ))}
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

function formatContextWindowCompact(tokens: number): string {
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(1)}M`
  if (tokens >= 1_000) return `${Math.round(tokens / 1000)}K`
  return `${tokens}`
}

function formatPricing(pricing: { input: number; output: number }): string {
  return `$${pricing.input.toFixed(2)}/$${pricing.output.toFixed(2)}`
}

function formatMemoryGiB(bytes: number): string {
  const gib = bytes / 1024 ** 3
  return `${gib.toFixed(gib >= 10 ? 1 : 2)} GiB`
}

function SlotCard({
  entry,
  modelConfig,
  localChoices,
  isLast,
}: {
  entry: SlotEntry
  modelConfig?: UseModelConfigResult
  localChoices: readonly LocalModelChoice[]
  isLast: boolean
}): ReactNode {
  const slotId = entry.slotId
  const catalogState = modelConfig ? catalogStateOf(modelConfig) : Option.none()
  const models = modelConfig ? catalogModelsOf(modelConfig) : Option.none()
  const slotsState = modelConfig
    ? Option.map(Result.value(modelConfig.slots), ({ state }) => state)
    : Option.none()
  const selected = Option.flatMap(
    Option.all({ catalog: catalogState, slots: slotsState }),
    ({ catalog, slots }) => selectedSlotModel(catalog, slots, slotId),
  )
  const transportFailed = modelConfig !== undefined && Result.isFailure(modelConfig.catalog)
  const transportHasSnapshot = modelConfig !== undefined && Option.isSome(Result.value(modelConfig.catalog))
  const loading = Option.match(catalogState, {
    onNone: () => modelConfig !== undefined && !transportFailed,
    onSome: (state) => ModelCatalogLifecycle.is(state, "loading"),
  })

  const effectiveModelKey = Option.match(selected, {
    onNone: () => "",
    onSome: ({ model }) => `${model.providerId}\0${model.providerModelId}`,
  })
  const effortControl = Option.match(selected, {
    onNone: () => ({ _tag: "Unavailable", label: "Unassigned" } as const),
    onSome: ({ model }) => reasoningEffortControl(model),
  })
  const effortOptions = effortControl._tag === "Available" ? effortControl.options : []
  const currentEffort = Option.match(selected, {
    onNone: () => "",
    onSome: ({ slot }) => slot.selection.reasoningEffort,
  })
  const capacityRisk = Option.flatMap(selected, ({ model }) => {
    if (model.providerId !== "local") return Option.none()
    const choice = localChoices.find((candidate) => candidate.providerModelId === model.providerModelId)
    if (choice?.fitAssessment._tag !== "Assessed" || choice.fitAssessment.result !== "does_not_fit") return Option.none()
    return Option.some(choice.fitAssessment)
  })

  const handleModelChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const [providerId, providerModelId] = e.target.value.split("\0")
      if (modelConfig) {
        if (!providerId || !providerModelId) {
          void modelConfig.updateSlotModel(slotId, null, null)
        } else {
          const model = Option.flatMap(models, (catalogModels) => Option.fromNullable(
            catalogModels.find((candidate) => candidate.providerId === providerId
              && candidate.providerModelId === providerModelId),
          ))
          if (Option.isSome(model)) {
            void modelConfig.updateSlotModel(slotId, model.value.providerId, model.value.providerModelId)
          }
        }
      }
    },
    [modelConfig, slotId, models],
  )

  const handleEffortChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      if (!modelConfig || Option.isNone(selected)) return
      const value = e.target.value as ReasoningEffort
      void modelConfig.updateSlotReasoning(
        slotId,
        value === selected.value.model.defaultReasoningEffort ? null : value,
      )
    },
    [modelConfig, selected, slotId],
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
          {entry.label}
        </span>
      </div>

      {(transportFailed && !transportHasSnapshot)
        || Option.exists(catalogState, (state) => ModelCatalogLifecycle.is(state, "unavailable")) ? (
        <div style={{ marginBottom: 4 }}>
          <span style={{ fontSize: 13, color: "var(--accent-error)" }}>
            Unable to load available models
          </span>
        </div>
      ) : loading && Option.isNone(models) ? (
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
            value={effectiveModelKey}
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
            {Option.isNone(selected) && <option value="" disabled>Unassigned</option>}
            {Option.isSome(models) && models.value.length > 0 ? (
              models.value.map((model) => (
                <option key={`${model.providerId}:${model.providerModelId}`} value={`${model.providerId}\0${model.providerModelId}`}>
                  {model.displayName} — {formatContextWindowCompact(model.contextWindow)} ctx
                  {model.pricing ? ` — ${formatPricing(model.pricing)}` : ""}
                  {model.providerId === "local" && localChoices.some((choice) => choice.providerModelId === model.providerModelId
                    && choice.fitAssessment._tag === "Assessed"
                    && choice.fitAssessment.result === "does_not_fit") ? " — memory warning" : ""}
                </option>
              ))
            ) : (
              <option value="">{entry.modelDisplayName}</option>
            )}
          </select>

          {/* Thinking level dropdown */}
          <select
            value={currentEffort}
            onChange={handleEffortChange}
            disabled={effortControl._tag === "Unavailable"}
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
            {effortControl._tag === "Unavailable" && (
              <option value={currentEffort}>{effortControl.label}</option>
            )}
            {effortOptions.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
          </div>
          {Option.isSome(capacityRisk) && (
            <div style={{ marginBottom: 6, fontSize: 12, color: "var(--accent-warning)" }}>
              Estimated memory use is {formatMemoryGiB(capacityRisk.value.requiredTotalBytes)}, above this machine&apos;s stable capacity. Loading may fail or affect system performance.
            </div>
          )}
          {Option.isSome(selected) && (
            <div style={{ marginBottom: 4, fontSize: 12, color: "var(--fg-tertiary)" }}>
              {visionPropertyLabel(selected.value.model)} · {reasoningPropertyLabel(selected.value.model)}
            </div>
          )}
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
