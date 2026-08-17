import { formatDecimalGigabytes, type HardwareMemoryDomainView } from '@magnitudedev/client-common'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../hooks/use-theme'
import { StackedBar } from './stacked-bar'

interface HardwareMemoryDomainProps {
  readonly domain: HardwareMemoryDomainView
  readonly width?: number
}

type CompleteHardwareMemoryDomain = HardwareMemoryDomainView & {
  readonly usedBytes: number
  readonly fixedBytes: number
  readonly kvCacheBytes: number
  readonly systemAndAppsBytes: number
  readonly freeBytes: number
}

const isComplete = (domain: HardwareMemoryDomainView): domain is CompleteHardwareMemoryDomain =>
  domain.fixedBytes !== null
  && domain.kvCacheBytes !== null
  && domain.systemAndAppsBytes !== null
  && domain.freeBytes !== null
  && domain.usedBytes !== null

export const HardwareMemoryDomain = ({ domain, width = 48 }: HardwareMemoryDomainProps) => {
  const theme = useTheme()
  const complete = isComplete(domain)
  const barSegments = complete
    ? [
        { value: domain.fixedBytes, color: theme.text.body },
        { value: domain.kvCacheBytes, color: theme.accent },
        { value: domain.systemAndAppsBytes, color: theme.status.warning },
      ]
    : domain.usedBytes !== null && domain.freeBytes !== null
      ? [
          { value: domain.usedBytes, color: theme.text.supporting },
        ]
      : []

  return (
    <box style={{ flexDirection: 'column', paddingTop: 1 }}>
      <text style={{ fg: theme.text.body }} attributes={TextAttributes.BOLD}>{domain.label}</text>
      <text style={{ fg: theme.text.body }}>
        {domain.usedBytes === null
          ? `${formatDecimalGigabytes(domain.totalBytes)} total`
          : `${formatDecimalGigabytes(domain.usedBytes)} / ${formatDecimalGigabytes(domain.totalBytes)} used`}
      </text>
      {barSegments.length > 0 && (
        <StackedBar
          segments={barSegments}
          total={domain.totalBytes}
          width={width}
          trackColor={theme.border.standard}
        />
      )}
      {complete ? (
        <box style={{ flexDirection: 'column' }}>
          <text style={{ fg: theme.text.body }}><span fg={theme.text.body}>■</span>{` Weights       ${formatDecimalGigabytes(domain.fixedBytes)}`}</text>
          <text style={{ fg: theme.text.body }}><span fg={theme.accent}>■</span>{` KV cache      ${formatDecimalGigabytes(domain.kvCacheBytes)}`}</text>
          <text style={{ fg: theme.text.body }}><span fg={theme.status.warning}>■</span>{` System & apps ${formatDecimalGigabytes(domain.systemAndAppsBytes)}`}</text>
          <text style={{ fg: theme.text.body }}><span fg={theme.border.standard}>□</span>{` Free          ${formatDecimalGigabytes(domain.freeBytes)}`}</text>
        </box>
      ) : domain.notice ? (
        <text style={{ fg: theme.text.supporting }}><span attributes={TextAttributes.DIM}>{domain.notice}</span></text>
      ) : null}
    </box>
  )
}
