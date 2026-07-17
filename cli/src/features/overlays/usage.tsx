import { memo, useCallback, useMemo, useState, useSyncExternalStore } from 'react'
import { subscribeAnimationTick, getAnimationTickSnapshot, useAgentClient, useSettingsState } from '@magnitudedev/client-common'
import { TextAttributes, type KeyEvent } from '@opentui/core'
import { useKeyboard } from '@opentui/react'
import { useTheme } from '../../hooks/use-theme'
import { Button } from '../../components/button'
import type { CloudUsageResponse, UsagePeriod } from '@magnitudedev/sdk'
import { Atom, Result, useAtomValue } from '@effect-atom/atom-react'
import { authSourceAtom } from '../../state/cli-atoms'
import { hasCloudUsageAuth } from './usage-auth'

interface UsageOverlayProps {
  isVisible: boolean
  onClose: () => void
}

const PERIODS: ReadonlyArray<{ id: UsagePeriod; label: string }> = [
  { id: '24h', label: '24h' },
  { id: '3d', label: '3d' },
  { id: '7d', label: '7d' },
  { id: '14d', label: '14d' },
  { id: '30d', label: '30d' },
  { id: 'all', label: 'all' },
]

const DAILY_DAYS = 30
const TOP_MODELS_COUNT = 5
const USAGE_LIMIT_BAR_WIDTH = 18

export function formatReset(remainingMs: number): string {
  const minutes = Math.max(0, Math.ceil(remainingMs / (60 * 1000)))
  if (minutes < 1) return '<1m'
  if (minutes < 60) return `${minutes}m`

  const hours = Math.floor(minutes / 60)
  const remainingMinutes = minutes % 60
  if (hours < 24) return remainingMinutes > 0 ? `${hours}h ${remainingMinutes}m` : `${hours}h`

  const days = Math.floor(hours / 24)
  const remainingHours = hours % 24
  return remainingHours > 0 ? `${days}d ${remainingHours}h` : `${days}d`
}

function formatTokens(n: number): string {
  if (n < 1000) return String(n)
  if (n < 1_000_000) return `${(n / 1000).toFixed(n < 10_000 ? 1 : 0)}k`
  return `${(n / 1_000_000).toFixed(1)}M`
}

export function usagePercentLeft(usedCents: number, limitCents: number): number {
  if (limitCents <= 0) return 0
  return Math.max(0, Math.min(100, Math.round((1 - usedCents / limitCents) * 100)))
}

function requestLabel(count: number): string {
  return `${count} ${count === 1 ? 'request' : 'requests'}`
}

export function formatUsageSummary(totals: {
  readonly requestCount: number
  readonly inputTokens: number
  readonly outputTokens: number
}): string {
  return `${requestLabel(totals.requestCount)}  ·  ${formatTokens(totals.inputTokens)} input tokens  ·  ${formatTokens(totals.outputTokens)} output tokens`
}

export function formatModelUsage(model: {
  readonly requestCount: number
  readonly inputTokens: number
  readonly outputTokens: number
}): string {
  return `${requestLabel(model.requestCount)}  ·  ${formatTokens(model.inputTokens)} in / ${formatTokens(model.outputTokens)} out`
}

function formatDate(iso: string): string {
  // YYYY-MM-DD → MM-DD
  return iso.slice(5)
}

function getLocalTimeZone(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone
  } catch {
    return 'UTC'
  }
}

interface TabBarProps {
  value: UsagePeriod
  onChange: (p: UsagePeriod) => void
}

function TabBar({ value, onChange }: TabBarProps) {
  const theme = useTheme()
  const [hovered, setHovered] = useState<UsagePeriod | null>(null)

  return (
    <box style={{ flexDirection: 'row', flexShrink: 0 }}>
      {PERIODS.map((p, idx) => {
        const isActive = p.id === value
        const isHovered = hovered === p.id
        const fg = isActive ? theme.primary : isHovered ? theme.foreground : theme.muted
        const attrs = isActive ? TextAttributes.BOLD | TextAttributes.UNDERLINE : undefined
        return (
          <box key={p.id} style={{ flexDirection: 'row' }}>
            {idx > 0 && <text style={{ fg: theme.border }}>{' · '}</text>}
            <Button
              onClick={() => onChange(p.id)}
              onMouseOver={() => setHovered(p.id)}
              onMouseOut={() => setHovered(prev => (prev === p.id ? null : prev))}
            >
              <text style={{ fg }} attributes={attrs}>{p.label}</text>
            </Button>
          </box>
        )
      })}
      <text style={{ fg: theme.muted }}>{'   '}</text>
      <text style={{ fg: theme.border }} attributes={TextAttributes.DIM}>
        Tab / Shift+Tab to switch
      </text>
    </box>
  )
}

interface DailyUsageRowProps {
  date: string
  inputTokens: number
  outputTokens: number
  topModel: string | null
}

function DailyUsageRow({ date, inputTokens, outputTokens, topModel }: DailyUsageRowProps) {
  const theme = useTheme()
  return (
    <box style={{ flexDirection: 'row' }}>
      <text style={{ fg: theme.muted }}>{formatDate(date)}{'  '}</text>
      <text style={{ fg: theme.foreground }}>
        {formatTokens(inputTokens)} in / {formatTokens(outputTokens)} out
      </text>
      {topModel && (
        <text style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>
          {`  · ${topModel}`}
        </text>
      )}
    </box>
  )
}

interface UsageLimitBarProps {
  label: string
  limitCents: number
  usedCents: number
  remainingMs: number
}

function UsageLimitBar({ label, limitCents, usedCents, remainingMs }: UsageLimitBarProps) {
  const theme = useTheme()
  const percentLeft = usagePercentLeft(usedCents, limitCents)
  const filled = Math.round(percentLeft / 100 * USAGE_LIMIT_BAR_WIDTH)
  const empty = USAGE_LIMIT_BAR_WIDTH - filled
  const activeColor = percentLeft === 0 ? theme.error : theme.primary

  return (
    <box style={{ flexDirection: 'row' }}>
      <text style={{ fg: theme.foreground }}>{label.padEnd(9)}</text>
      <text>
        <span fg={activeColor}>{'▇'.repeat(filled)}</span>
        <span fg={theme.border}>{'·'.repeat(empty)}</span>
      </text>
      <text style={{ fg: percentLeft === 0 ? theme.error : theme.foreground }}>
        {`  ${String(percentLeft).padStart(3)}% left`}
      </text>
      <text style={{ fg: theme.muted }}>
        {` · resets in ${formatReset(remainingMs)}`}
      </text>
    </box>
  )
}

export const UsageOverlay = memo(function UsageOverlay({ isVisible, onClose }: UsageOverlayProps) {
  const theme = useTheme()
  const client = useAgentClient()
  const settings = useSettingsState()
  const authSource = useAtomValue(authSourceAtom)
  const runtimeResult = useAtomValue(client.runtime)
  const [period, setPeriod] = useState<UsagePeriod>('7d')

  const tz = useMemo(getLocalTimeZone, [])
  const runtimeReady = Result.isSuccess(runtimeResult)
  const cloudConfigured = hasCloudUsageAuth(settings.keyAlreadySet, authSource)

  const usageAtom = useMemo(
    () => isVisible && runtimeReady && cloudConfigured
      ? client.query('GetCloudUsage', { period, days: DAILY_DAYS, tz })
      : Atom.make<Result.Result<CloudUsageResponse, never>>(() => Result.initial()),
    [client, cloudConfigured, isVisible, period, runtimeReady, tz],
  )
  const result = useAtomValue(usageAtom)

  const loading = Result.isInitial(result)
  const error = Result.isFailure(result) ? 'Failed to load usage data' : null
  const data = Result.isSuccess(result) ? result.value.data : null

  // Loading dots animation via tick store (400ms ≈ 5 ticks per step)
  const tick = useSyncExternalStore(subscribeAnimationTick, getAnimationTickSnapshot, getAnimationTickSnapshot)
  const loadingTick = isVisible && cloudConfigured && data === null ? tick : 0

  const periodIndex = PERIODS.findIndex(p => p.id === period)

  const onKey = useCallback((key: KeyEvent) => {
    if (!isVisible) return
    if (key.name === 'escape') {
      key.preventDefault()
      onClose()
      return
    }
    if (key.name === 'tab') {
      key.preventDefault()
      const delta = key.shift ? -1 : 1
      const next = (periodIndex + delta + PERIODS.length) % PERIODS.length
      setPeriod(PERIODS[next].id)
    }
  }, [isVisible, periodIndex, onClose])

  useKeyboard(onKey)

  if (!isVisible) return null

  return (
    <box style={{ flexDirection: 'column', height: '100%' }}>
      {/* Header */}
      <box style={{ flexDirection: 'row', paddingLeft: 2, paddingRight: 2, paddingTop: 1, paddingBottom: 1, flexShrink: 0 }}>
        <text style={{ fg: theme.primary, flexGrow: 1 }}>
          <span attributes={TextAttributes.BOLD}>Usage</span>
        </text>
        <text style={{ fg: theme.muted }}>
          <span attributes={TextAttributes.DIM}>Esc to close</span>
        </text>
      </box>

      <box style={{ paddingLeft: 2, paddingRight: 2, flexShrink: 0 }}>
        <text style={{ fg: theme.border }}>{'─'.repeat(60)}</text>
      </box>

      {/* Body */}
      <box style={{ paddingLeft: 2, paddingRight: 2, paddingTop: 1, paddingBottom: 1, flexDirection: 'column', flexGrow: 1 }}>
        {!cloudConfigured && (
          <text style={{ fg: theme.muted }}>
            Connect cloud models in /settings to view cloud usage.
          </text>
        )}
        {cloudConfigured && error && (
          <text style={{ fg: theme.error }}>Failed to load usage: {error}</text>
        )}
        {cloudConfigured && !error && !data && (
          <text style={{ fg: theme.muted }}>
            <span attributes={TextAttributes.DIM}>Loading{'.'.repeat((loadingTick % 3) + 1)}</span>
          </text>
        )}
        {cloudConfigured && !error && data && (
          <UsageBody
            data={data}
            period={period}
            onPeriodChange={setPeriod}
            loading={loading}
          />
        )}
      </box>
    </box>
  )
})

interface UsageBodyProps {
  data: CloudUsageResponse['data']
  period: UsagePeriod
  onPeriodChange: (p: UsagePeriod) => void
  loading: boolean
}

function UsageBody({ data, period, onPeriodChange, loading }: UsageBodyProps) {
  const theme = useTheme()
  const { subscription, usageWindows, usage } = data

  const topModels = usage.byModel.slice(0, TOP_MODELS_COUNT)
  const usageLimitRows = [
    { key: 'five_hour', label: '5-hour' },
    { key: 'weekly', label: 'Weekly' },
    { key: 'monthly', label: 'Monthly' },
  ] as const

  return (
    <box style={{ flexDirection: 'column' }}>
      {/* Cloud subscription and current limit windows */}
      <box style={{ flexDirection: 'column', paddingBottom: 1 }}>
        <text style={{ fg: theme.foreground }}>
          <span attributes={TextAttributes.BOLD}>Cloud plan: </span>
          <span fg={subscription.status === 'active' ? theme.primary : theme.muted}>
            {subscription.status === 'active' ? subscription.plan.label : 'Not subscribed'}
          </span>
        </text>
        {subscription.status !== 'active' && (
          <text style={{ fg: theme.muted }}>
            Magnitude Pro is required to use cloud models.
          </text>
        )}
        {subscription.status === 'active' && (
          <box style={{ flexDirection: 'column', paddingTop: 1 }}>
            <text style={{ fg: theme.foreground }}>
              <span attributes={TextAttributes.BOLD}>Usage limits</span>
            </text>
            {usageLimitRows.map(({ key, label }) => {
              const budget = usageWindows[key]
              return budget ? (
                <UsageLimitBar
                  key={key}
                  label={label}
                  limitCents={budget.limitCents}
                  usedCents={budget.usedCents}
                  remainingMs={budget.remainingMs}
                />
              ) : null
            })}
          </box>
        )}
      </box>

      <box style={{ paddingBottom: 1 }}>
        <text style={{ fg: theme.border }}>{'─'.repeat(60)}</text>
      </box>

      {/* Period tabs */}
      <box style={{ paddingBottom: 1 }}>
        <TabBar value={period} onChange={onPeriodChange} />
      </box>

      {/* Period summary */}
      <box style={{ flexDirection: 'row', paddingBottom: 1 }}>
        <text style={{ fg: loading ? theme.muted : theme.foreground }}>
          {formatUsageSummary(usage.totals)}
        </text>
        {loading && (
          <text style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>{'  (loading…)'}</text>
        )}
      </box>

      {/* Top models */}
      <box style={{ flexDirection: 'column', paddingBottom: 1 }}>
        <text style={{ fg: theme.foreground }}>
          <span attributes={TextAttributes.BOLD}>Top models</span>
        </text>
        {topModels.length === 0 && (
          <text style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>No usage in this period</text>
        )}
        {topModels.map(m => (
          <box key={m.model} style={{ flexDirection: 'row' }}>
            <text style={{ fg: theme.foreground }}>{m.model.padEnd(28).slice(0, 28)}</text>
            <text style={{ fg: theme.muted }}>{'  '}</text>
            <text style={{ fg: theme.foreground }}>{formatModelUsage(m)}</text>
          </box>
        ))}
      </box>

      {/* Daily chart */}
      <box style={{ flexDirection: 'column' }}>
        <text style={{ fg: theme.foreground }}>
          <span attributes={TextAttributes.BOLD}>Daily tokens (last {DAILY_DAYS} days)</span>
        </text>
        {usage.dailyTokens.length === 0 && (
          <text style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>No daily activity</text>
        )}
        {usage.dailyTokens.map(d => (
          <DailyUsageRow
            key={d.date}
            date={d.date}
            inputTokens={d.inputTokens}
            outputTokens={d.outputTokens}
            topModel={d.topModel}
          />
        ))}
      </box>
    </box>
  )
}

export type { UsageOverlayProps }
