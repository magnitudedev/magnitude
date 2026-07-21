import type { HardwareMemoryDomainView } from '@magnitudedev/client-common'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../hooks/use-theme'
import { StackedBar } from './stacked-bar'

const formatMemoryBytes = (bytes: number): string => {
  const gib = bytes / 1024 ** 3
  return gib >= 1 ? `${gib.toFixed(1)} GiB` : `${Math.round(bytes / 1024 ** 2)} MiB`
}

interface HardwareMemoryDomainProps {
  readonly domain: HardwareMemoryDomainView
  readonly width?: number
}

export const HardwareMemoryDomain = ({ domain, width = 48 }: HardwareMemoryDomainProps) => {
  const theme = useTheme()
  const complete = domain.fixedBytes !== null
    && domain.kvCacheBytes !== null
    && domain.systemAndAppsBytes !== null
    && domain.freeBytes !== null
    && domain.usedBytes !== null
  const barSegments = complete
    ? [
        { value: domain.fixedBytes!, color: theme.foreground },
        { value: domain.kvCacheBytes!, color: theme.primary },
        { value: domain.systemAndAppsBytes!, color: theme.warning },
      ]
    : domain.usedBytes !== null && domain.freeBytes !== null
      ? [
          { value: domain.usedBytes, color: theme.secondary },
        ]
      : []

  return (
    <box style={{ flexDirection: 'column', paddingTop: 1 }}>
      <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>{domain.label}</text>
      <text style={{ fg: theme.foreground }}>
        {domain.usedBytes === null
          ? `${formatMemoryBytes(domain.totalBytes)} total`
          : `${formatMemoryBytes(domain.usedBytes)} / ${formatMemoryBytes(domain.totalBytes)} used`}
      </text>
      {barSegments.length > 0 && (
        <StackedBar
          segments={barSegments}
          total={domain.totalBytes}
          width={width}
          trackColor={theme.border}
        />
      )}
      {complete ? (
        <box style={{ flexDirection: 'column' }}>
          <text><span fg={theme.foreground}>■</span>{` Weights       ${formatMemoryBytes(domain.fixedBytes!)}`}</text>
          <text><span fg={theme.primary}>■</span>{` KV cache      ${formatMemoryBytes(domain.kvCacheBytes!)}`}</text>
          <text><span fg={theme.warning}>■</span>{` System & apps ${formatMemoryBytes(domain.systemAndAppsBytes!)}`}</text>
          <text><span fg={theme.border}>□</span>{` Free          ${formatMemoryBytes(domain.freeBytes!)}`}</text>
        </box>
      ) : domain.notice ? (
        <text style={{ fg: theme.muted }}><span attributes={TextAttributes.DIM}>{domain.notice}</span></text>
      ) : null}
    </box>
  )
}

export { formatMemoryBytes }
