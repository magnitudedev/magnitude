import { deriveHardwareMemoryView, type LocalInferenceState } from '@magnitudedev/client-common'
import { TextAttributes } from '@opentui/core'
import { Option } from 'effect'
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
  const progress = Option.getOrNull(operation.progress)
  if (operation.kind === 'download' && progress && 'totalBytes' in progress && progress.totalBytes > 0) {
    const percent = Math.round(progress.completedBytes / progress.totalBytes * 100)
    return `Downloading ${percent}%`
  }
  if (operation.kind === 'activate' && progress && 'fraction' in progress) return `Loading ${Math.round(progress.fraction * 100)}%`
  const action = operation.kind === 'activate' ? 'Loading' : 'Downloading'
  return `${action} · ${operation.stage.replaceAll('_', ' ')}`
}

const compactMemory = (bytes: number): string => formatMemoryBytes(bytes).replace('.0 ', ' ')

const compactBarSegments = (
  domains: ReturnType<typeof deriveHardwareMemoryView>['domains'],
  colors: { readonly fixed: string; readonly kv: string; readonly system: string },
): readonly StackedBarSegment[] | null => {
  const participating = domains.filter((domain) => domain.participatesInRuntime && domain.usedBytes !== null)
  if (participating.length === 0) return null
  const complete = participating.filter((domain): domain is typeof domain & {
    readonly fixedBytes: number
    readonly kvCacheBytes: number
    readonly systemAndAppsBytes: number
  } =>
    domain.fixedBytes !== null
    && domain.kvCacheBytes !== null
    && domain.systemAndAppsBytes !== null)
  if (complete.length !== participating.length) {
    return [{
      value: participating.reduce((sum, domain) => sum + (domain.usedBytes ?? 0), 0),
      color: colors.system,
    }]
  }
  return [
    {
      value: complete.reduce((sum, domain) => sum + domain.fixedBytes, 0),
      color: colors.fixed,
    },
    {
      value: complete.reduce((sum, domain) => sum + domain.kvCacheBytes, 0),
      color: colors.kv,
    },
    {
      value: complete.reduce((sum, domain) => sum + domain.systemAndAppsBytes, 0),
      color: colors.system,
    },
  ]
}

export const LocalRuntimeStatusBar = ({ state, width, onOpenHardware }: LocalRuntimeStatusBarProps) => {
  const theme = useTheme()
  const operation = Option.fromNullable(state.operations.at(-1))
  const running = Option.fromNullable(state.choices.find((choice) => choice._tag === 'Running'))
  if (Option.isNone(operation) && Option.isNone(running)) return null

  const choice = Option.orElse(
    Option.flatMap(operation, (current) => Option.fromNullable(
      state.choices.find((candidate) => candidate.providerModelId === current.providerModelId),
    )),
    () => running,
  )
  const modelName = Option.getOrElse(
    Option.map(choice, (selected) => selected.displayName),
    () => Option.getOrElse(
      Option.map(operation, (current) => current.providerModelId),
      () => 'Local model',
    ),
  )
  const status = Option.match(operation, { onNone: () => 'Ready', onSome: operationLabel })
  const assessedDomainIds = Option.match(choice, {
    onNone: () => [],
    onSome: (selected) => selected.fitAssessment._tag === 'Assessed'
      ? selected.fitAssessment.domains.map((domain) => domain.memoryDomainId)
      : [],
  })
  const memoryView = deriveHardwareMemoryView(state.host, {
    participatingDomainIds: assessedDomainIds,
    fallbackToAccelerators: Option.isSome(operation),
  })
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
