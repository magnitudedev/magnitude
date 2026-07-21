import { deriveHardwareMemoryView } from '@magnitudedev/client-common'
import type { LocalInferenceState } from '@magnitudedev/sdk'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../../hooks/use-theme'
import { Button } from '../../components/button'
import { formatMemoryBytes } from '../../components/hardware-memory-domain'
import { StackedBar, type StackedBarSegment } from '../../components/stacked-bar'

interface LocalRuntimeStatusBarProps {
  readonly state: LocalInferenceState
  readonly width: number
  readonly onOpenHardware: () => void
}

const operationLabel = (operation: LocalInferenceState['operations'][number]): string => {
  if (operation.status === 'failed') {
    return `Failed · ${operation.stage.replaceAll('_', ' ')}`
  }
  if (operation.kind === 'download' && operation.progress && operation.progress.totalBytes > 0) {
    const percent = Math.round(operation.progress.completedBytes / operation.progress.totalBytes * 100)
    return `Downloading ${percent}%`
  }
  const action = operation.kind === 'restart' ? 'Restarting' : operation.kind === 'activate' ? 'Loading' : 'Downloading'
  return `${action} · ${operation.stage.replaceAll('_', ' ')}`
}

const compactMemory = (bytes: number): string => formatMemoryBytes(bytes).replace('.0 ', ' ')

const compactBarSegments = (
  domains: ReturnType<typeof deriveHardwareMemoryView>['domains'],
  colors: { readonly fixed: string; readonly kv: string; readonly system: string },
): readonly StackedBarSegment[] | null => {
  const participating = domains.filter((domain) => domain.participatesInRuntime && domain.usedBytes !== null)
  if (participating.length === 0) return null
  const complete = participating.every((domain) =>
    domain.fixedBytes !== null
    && domain.kvCacheBytes !== null
    && domain.systemAndAppsBytes !== null)
  if (!complete) {
    return [{
      value: participating.reduce((sum, domain) => sum + (domain.usedBytes ?? 0), 0),
      color: colors.system,
    }]
  }
  return [
    {
      value: participating.reduce((sum, domain) => sum + domain.fixedBytes!, 0),
      color: colors.fixed,
    },
    {
      value: participating.reduce((sum, domain) => sum + domain.kvCacheBytes!, 0),
      color: colors.kv,
    },
    {
      value: participating.reduce((sum, domain) => sum + domain.systemAndAppsBytes!, 0),
      color: colors.system,
    },
  ]
}

export const LocalRuntimeStatusBar = ({ state, width, onOpenHardware }: LocalRuntimeStatusBarProps) => {
  const theme = useTheme()
  const latestOperation = state.operations.at(-1)
  const operation = latestOperation?.status === 'completed' ? undefined : latestOperation
  const running = state.choices.find((choice) => choice._tag === 'Running')
  if (!operation && !running) return null

  const choice = operation
    ? state.choices.find((candidate) => candidate.providerModelId === operation.providerModelId)
    : running
  const modelName = choice?.displayName ?? operation?.providerModelId ?? running?.displayName ?? 'Local model'
  const status = operation ? operationLabel(operation) : 'Ready'
  const assessedDomainIds = choice?.fitAssessment._tag === 'Assessed'
    ? choice.fitAssessment.domains.map((domain) => domain.memoryDomainId)
    : []
  const memoryView = state.host._tag === 'Available'
    ? deriveHardwareMemoryView(state.host.profile, {
        participatingDomainIds: assessedDomainIds,
        fallbackToAccelerators: operation !== undefined,
      })
    : null
  const memory = memoryView?.compact ?? null
  const barSegments = memoryView
    ? compactBarSegments(memoryView.domains, {
        fixed: theme.foreground,
        kv: theme.primary,
        system: theme.warning,
      })
    : null
  const barWidth = width >= 84 ? 16 : width >= 72 ? 12 : 0
  const showMemoryWord = width >= 62

  return (
    <box style={{
      marginLeft: 1,
      marginRight: 1,
      flexDirection: 'row',
      flexShrink: 0,
      borderStyle: 'rounded',
      borderColor: theme.border,
      paddingLeft: 1,
      paddingRight: 1,
    }}>
      <text style={{ fg: theme.foreground, flexShrink: 1 }} attributes={TextAttributes.BOLD}>{modelName}</text>
      <text style={{ fg: theme.muted }}>  {status}</text>
      <box style={{ flexGrow: 1 }} />
      {memory && (
        <Button onClick={onOpenHardware} style={{ flexDirection: 'row' }}>
          {barSegments && barWidth > 0 && (
            <>
              <StackedBar
                segments={barSegments}
                total={memory.totalBytes}
                width={barWidth}
                trackColor={theme.border}
              />
              <text> </text>
            </>
          )}
          <text style={{ fg: theme.link }}>
            {showMemoryWord ? 'Memory ' : ''}{compactMemory(memory.usedBytes)} / {compactMemory(memory.totalBytes)}
          </text>
        </Button>
      )}
    </box>
  )
}
