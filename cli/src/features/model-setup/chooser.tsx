import { Fragment, useCallback, useMemo, useRef, useState, useSyncExternalStore, type ReactNode, type Ref } from "react"
import { TextAttributes, type KeyEvent, type ScrollBoxRenderable } from "@opentui/core"
import { useKeyboard } from "@opentui/react"
import { Result } from "@effect-atom/atom-react"
import { Option } from "effect"
import {
  getAnimationTimeSnapshot,
  truncateToDisplayWidth,
  formatLocalModelDisplayName,
  formatMemorySize,
  type OnboardingModelLoadStatus,
  type LocalModelOption,
  type LocalInferenceHardwareResult,
  type OnboardingModelRankingControls,
  LOCAL_MODEL_RANKING_SCALE_INTERVALS,
  LOCAL_MODEL_RANKING_SCALE_LABELS,
  rankedLocalModelOptions,
  targetPhysicalMemoryBytes,
  wrapTextToWordLines,
} from "@magnitudedev/client-common"
import type {
  LocalModel,
  LocalModelMemory,
  LocalModelDiscoveryProgressStep,
  ProviderModelId,
} from "@magnitudedev/sdk"
import { Button } from "../../components/button"
import {
  PENTAGON_RADAR_COLUMNS,
  PENTAGON_RADAR_ROWS,
  pentagonRadarValues,
  retargetPentagonRadar,
  type PentagonRadarTransition,
} from "../../components/pentagon-radar"
import { PentagonRadarView } from "../../components/pentagon-radar-view"
import { spinnerFrameAt, useSpinnerFrame } from "../../hooks/use-spinner-frame"
import { useTheme } from "../../hooks/use-theme"
import { BOX_CHARS } from "../../utils/ui-constants"
import { subscribeScrollboxActivity } from "../../utils/scroll-helpers"
import {
  describeLocalHardwareSummary,
  localInferenceProgressLines,
  selectedInferenceIndex,
  selectionContextLabel,
  selectionModelId,
  selectionProviderModelId,
  type LocalInferenceSelection,
} from "../local-inference/view-model"
import { localModelRadarAxes } from "@magnitudedev/client-common"
import {
  formatModelClassification,
  formatModelReleaseRecency,
} from "../local-inference/model-classification"
import { discoveredModelLocation } from "./discovered-model"
import {
  OnboardingModelConfiguringProgress,
  OnboardingModelDownloadProgress,
  OnboardingModelLoadProgress,
} from "./model-status"
import { isWideSetupLayout, SetupFrame, setupBodyWidth, type SetupStage } from "./setup-frame"

const SECTION_VIEWPORT_ROWS = 4
const RECOMMENDED_VIEWPORT_ROWS = 10
const MODEL_SECTION_HEADING_ROWS = 1
const MODEL_TITLE_ROWS = 1
const MODEL_SUMMARY_ROWS = 1
const MODEL_SUMMARY_RADAR_GAP_ROWS = 1
const WIDE_LIST_WIDTH = 38
const RANKING_LABEL_GAP = 2
const FAST_TO_SMART_SEGMENT_COLUMNS = Math.max(...LOCAL_MODEL_RANKING_SCALE_LABELS
  .slice(1)
  .map((label, index) => Math.ceil(LOCAL_MODEL_RANKING_SCALE_LABELS[index]!.length / 2)
    + Math.floor(label.length / 2)
    + RANKING_LABEL_GAP))
const FAST_TO_SMART_TRACK_COLUMNS = FAST_TO_SMART_SEGMENT_COLUMNS
  * LOCAL_MODEL_RANKING_SCALE_INTERVALS
const FAST_TO_SMART_TRACK_LEFT_PADDING = Math.floor(LOCAL_MODEL_RANKING_SCALE_LABELS[0].length / 2)
const FAST_TO_SMART_LABEL_LAYOUT = (() => {
  let previousEnd = 0
  return LOCAL_MODEL_RANKING_SCALE_LABELS.map((label, index) => {
    const center = FAST_TO_SMART_TRACK_LEFT_PADDING + index * FAST_TO_SMART_SEGMENT_COLUMNS
    const start = center - Math.floor(label.length / 2)
    const leadingSpaces = start - previousEnd
    previousEnd = start + label.length
    return { label, leadingSpaces }
  })
})()
export const ONBOARDING_RANKING_CONTROL_ROWS = 3

const onboardingModelRowId = (selectionId: string): string =>
  `onboarding-model:${selectionId}`

export const scrollOnboardingModelIntoView = (
  scrollbox: Pick<ScrollBoxRenderable, "scrollChildIntoView"> | null,
  selectionId: string,
): void => {
  scrollbox?.scrollChildIntoView(onboardingModelRowId(selectionId))
}

const scrollOnboardingModelPastOverflowIndicators = (
  scrollbox: ScrollBoxRenderable | null,
  selectionId: string,
  itemIndex: number,
  itemCount: number,
  viewportRows: number,
): void => {
  if (scrollbox === null) return
  scrollOnboardingModelIntoView(scrollbox, selectionId)
  const scrollTop = Math.max(0, Math.floor(scrollbox.scrollTop + 0.001))
  if (scrollTop > 0 && itemIndex === scrollTop) {
    scrollbox.scrollTo(Math.max(0, scrollTop - 1))
    return
  }
  if (scrollTop + viewportRows < itemCount && itemIndex === scrollTop + viewportRows - 1) {
    scrollbox.scrollTo(scrollTop + 1)
  }
}

export const onboardingModelActionLabel = (selection: LocalInferenceSelection): string => {
  if (selection.kind === "running") return "Loaded"
  if (selection.kind === "downloadable") return "Download"
  return "Load"
}

const onboardingSelection = (
  selection: LocalInferenceSelection,
): ProviderModelId | null => Option.getOrNull(selectionProviderModelId(selection))

export const onboardingModelRowName = (
  selection: LocalInferenceSelection,
): string => formatLocalModelDisplayName(selection.model)

const matchesOnboardingSelection = (
  selection: LocalInferenceSelection,
  submitted: ProviderModelId,
): boolean => Option.contains(selectionProviderModelId(selection), submitted)

const ModelRow = ({
  selection,
  selected,
  disabled,
  width,
  rowId,
  rank,
  onHover,
  onChoose,
}: {
  readonly selection: LocalInferenceSelection
  readonly selected: boolean
  readonly disabled: boolean
  readonly width: number
  readonly rowId: string
  readonly rank?: number
  readonly onHover: () => void
  readonly onChoose: () => void
}): ReactNode => {
  const theme = useTheme()
  const action = onboardingModelActionLabel(selection)
  const enabled = true
  const markerWidth = 2
  const rankLabel = rank === undefined ? "" : `${rank}. `
  const gap = 2
  const nameWidth = Math.max(1, width - markerWidth - rankLabel.length - gap - action.length - 1)
  return (
    <Button
      id={rowId}
      onClick={() => { if (enabled && !disabled) onChoose() }}
      onMouseOver={() => { if (!disabled) onHover() }}
      cursor={enabled && !disabled ? "pointer" : "default"}
      style={{ width: "100%", flexDirection: "row" }}
    >
      <text
        style={{ fg: selected ? theme.accent : enabled ? theme.text.body : theme.text.disabled }}
        attributes={selected ? TextAttributes.BOLD : TextAttributes.NONE}
        wrapMode="none"
      >
        {selected ? "› " : "  "}
        {rankLabel.length > 0 && <span fg={theme.text.detail}>{rankLabel}</span>}
        {truncateToDisplayWidth(onboardingModelRowName(selection), nameWidth).padEnd(nameWidth)}
        {"  "}
        <span fg={selection.kind === "running"
          ? theme.status.success
          : selection.kind === "downloadable" || selected
            ? theme.accent
            : theme.text.supporting}>
          {action}
        </span>
      </text>
    </Button>
  )
}

const ModelSectionViewport = ({
  scrollRef,
  rows,
  itemCount,
  showOverflow,
  children,
}: {
  readonly scrollRef: Ref<ScrollBoxRenderable | null>
  readonly rows: number
  readonly itemCount: number
  readonly showOverflow: boolean
  readonly children: ReactNode
}): ReactNode => {
  const theme = useTheme()
  const mountedScrollboxRef = useRef<ScrollBoxRenderable | null>(null)
  const attachScrollbox = useCallback((value: ScrollBoxRenderable | null) => {
    mountedScrollboxRef.current = value
    if (typeof scrollRef === "function") scrollRef(value)
    else if (scrollRef !== null) scrollRef.current = value
  }, [scrollRef])
  const subscribe = useCallback(
    (onStoreChange: () => void) => subscribeScrollboxActivity(
      mountedScrollboxRef.current,
      () => onStoreChange(),
    ),
    [],
  )
  const getScrollTop = useCallback(
    () => Math.max(0, mountedScrollboxRef.current?.scrollTop ?? 0),
    [],
  )
  const scrollTop = useSyncExternalStore(subscribe, getScrollTop, () => 0)
  const nativeHiddenAbove = showOverflow ? Math.min(itemCount, Math.floor(scrollTop + 0.001)) : 0
  const nativeHiddenBelow = showOverflow
    ? Math.max(0, itemCount - Math.ceil(scrollTop + rows - 0.001))
    : 0
  const hiddenAbove = nativeHiddenAbove > 0 ? nativeHiddenAbove + 1 : 0
  const hiddenBelow = nativeHiddenBelow > 0 ? nativeHiddenBelow + 1 : 0
  const overflowLine = (count: number) => (
    <text style={{
      fg: theme.text.disabled,
      height: 1,
      minHeight: 1,
      maxHeight: 1,
      flexShrink: 0,
    }} wrapMode="none">
      {count > 0 ? `  … and ${count} more` : ""}
    </text>
  )
  return (
    <box style={{
      position: "relative",
      flexDirection: "column",
      flexShrink: 0,
      height: rows,
      minHeight: rows,
      maxHeight: rows,
    }}>
      <scrollbox
        ref={attachScrollbox}
        focusable={false}
        scrollX={false}
        scrollbarOptions={{ visible: false }}
        style={{
          flexShrink: 0,
          rootOptions: {
            height: rows,
            minHeight: rows,
            maxHeight: rows,
            flexShrink: 0,
            backgroundColor: "transparent",
          },
          wrapperOptions: { border: false, backgroundColor: "transparent" },
          viewportOptions: { backgroundColor: "transparent" },
          contentOptions: { flexDirection: "column" },
        }}
      >
        {children}
      </scrollbox>
      {hiddenAbove > 0 && (
        <box style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          height: 1,
          zIndex: 100,
          backgroundColor: theme.background.terminal,
        }}>
          {overflowLine(hiddenAbove)}
        </box>
      )}
      {hiddenBelow > 0 && (
        <box style={{
          position: "absolute",
          bottom: 0,
          left: 0,
          right: 0,
          height: 1,
          zIndex: 100,
          backgroundColor: theme.background.terminal,
        }}>
          {overflowLine(hiddenBelow)}
        </box>
      )}
    </box>
  )
}

const DetailRow = ({
  width,
  children,
}: {
  readonly width: number
  readonly children?: ReactNode
}): ReactNode => (
  <box style={{
    width,
    height: 1,
    minHeight: 1,
    maxHeight: 1,
    flexShrink: 0,
    flexDirection: "row",
    overflow: "hidden",
  }}>
    {children}
  </box>
)

const OnboardingHardwareContext = ({
  hardware,
  width,
  spinnerFrame,
}: {
  readonly hardware: LocalInferenceHardwareResult
  readonly width: number
  readonly spinnerFrame: string
}): ReactNode => {
  const theme = useTheme()
  if (Result.isSuccess(hardware)) return describeLocalHardwareSummary(hardware.value).map((row) => (
    <text key={`${row.name}:${row.details.join(":")}`} style={{ width, flexShrink: 0 }} wrapMode="word">
      <span fg={theme.text.detail}>{row.name}</span>
      <span fg={theme.text.supporting}>{` · ${row.details.join(" · ")}`}</span>
    </text>
  ))
  if (Result.isFailure(hardware)) {
    return <text style={{ fg: theme.status.failure, width, flexShrink: 0 }}>! Hardware detection failed</text>
  }
  return (
    <text style={{ width, flexShrink: 0 }}>
      <span fg={theme.accent}>{spinnerFrame} </span>
      <span fg={theme.text.detail}>Detecting hardware…</span>
    </text>
  )
}

const hardwareSummaryText = ({
  name,
  details,
}: ReturnType<typeof describeLocalHardwareSummary>[number]): string =>
  `${name} · ${details.join(" · ")}`

export const onboardingHardwareContextRows = (
  hardware: LocalInferenceHardwareResult,
  width: number,
): number => {
  const bodyWidth = setupBodyWidth(width)
  if (Result.isSuccess(hardware)) {
    return describeLocalHardwareSummary(hardware.value).reduce(
      (rows, summary) => rows + wrapTextToWordLines(hardwareSummaryText(summary), bodyWidth).length,
      0,
    )
  }
  const message = Result.isFailure(hardware)
    ? "! Hardware detection failed"
    : `${spinnerFrameAt(0)} Detecting hardware…`
  return wrapTextToWordLines(message, bodyWidth).length
}

export const onboardingSetupAdditionalRows = (
  hardware: LocalInferenceHardwareResult,
  width: number,
): number => ONBOARDING_RANKING_CONTROL_ROWS
  + Math.max(0, onboardingHardwareContextRows(hardware, width) - 1)

const OnboardingSetupCard = ({
  width,
  stage,
  title,
  hardware,
  spinnerFrame = spinnerFrameAt(0),
  children,
  footer,
  unexpectedError,
}: {
  readonly width: number
  readonly stage: SetupStage
  readonly title?: string
  readonly hardware: LocalInferenceHardwareResult
  readonly spinnerFrame?: string
  readonly children: ReactNode
  readonly footer?: ReactNode
  readonly unexpectedError?: string | null
}): ReactNode => {
  const theme = useTheme()
  const bodyWidth = setupBodyWidth(width)
  const additionalRows = onboardingSetupAdditionalRows(hardware, width)
  return (
    <SetupFrame
      width={width}
      stage={stage}
      footer={footer}
      unexpectedError={unexpectedError}
      additionalRows={additionalRows}
    >
      {title !== undefined && (
        <text style={{ fg: theme.text.body }} attributes={TextAttributes.BOLD}>{title}</text>
      )}
      <OnboardingHardwareContext
        hardware={hardware}
        width={bodyWidth}
        spinnerFrame={spinnerFrame}
      />
      <box style={{ height: 1 }} />
      {children}
    </SetupFrame>
  )
}

export type OnboardingModelChooserOperation =
  | {
      readonly _tag: "Downloading"
      readonly model: LocalModel
      readonly starting: boolean
      readonly cancelling: boolean
      readonly onCancel: () => void
    }
  | {
      readonly _tag: "Configuring"
      readonly model: LocalModel
    }
  | {
      readonly _tag: "Activating"
      readonly providerModelId: ProviderModelId
      readonly model: LocalModel
      readonly status: OnboardingModelLoadStatus
      readonly onCancel: () => void
      readonly onRetry: () => void
      readonly onChooseAnother: () => void
    }

export const onboardingSelectionEnterAction = (
  kind: LocalInferenceSelection["kind"] | undefined,
): "download" | "load" | "select" | null => {
  if (kind === "downloadable") return "download"
  if (kind === "stored") return "load"
  if (kind === "running") return "select"
  return null
}

export const onboardingModelDetailRows = ({
  memoryWarning,
  modelSummaryRadarGap,
}: {
  readonly memoryWarning: boolean
  readonly modelSummaryRadarGap: boolean
}): number => MODEL_TITLE_ROWS
  + MODEL_SUMMARY_ROWS
  + (modelSummaryRadarGap ? MODEL_SUMMARY_RADAR_GAP_ROWS : 0)
  + PENTAGON_RADAR_ROWS
  + (memoryWarning ? 1 : 0)

export const ONBOARDING_MODEL_DETAIL_ROWS = onboardingModelDetailRows({
  memoryWarning: false,
  modelSummaryRadarGap: true,
})

export const onboardingLocalModelLayout = ({
  wide,
  localCount,
  detailPanelRows,
  rankedRows,
  sectionGap,
}: {
  readonly wide: boolean
  readonly localCount: number
  readonly detailPanelRows: number
  readonly rankedRows: number
  readonly sectionGap: number
}): { readonly viewportRows: number; readonly showOverflow: boolean } => {
  if (localCount === 0) return { viewportRows: 0, showOverflow: false }
  const availableRows = wide
    ? Math.max(1, detailPanelRows - rankedRows - sectionGap - MODEL_SECTION_HEADING_ROWS)
    : Math.min(SECTION_VIEWPORT_ROWS, localCount)
  const showOverflow = localCount > availableRows
  const viewportRows = wide
    ? Math.min(localCount, availableRows)
    : Math.min(SECTION_VIEWPORT_ROWS, localCount)
  return { viewportRows, showOverflow }
}

export function OnboardingModelChooser({
  hardware,
  options,
  rankingControls,
  onRankingControlsChange,
  width,
  error,
  operation,
  onSelect,
  onExit,
  exitKind,
}: {
  readonly hardware: LocalInferenceHardwareResult
  readonly options: readonly LocalModelOption[]
  readonly rankingControls: OnboardingModelRankingControls
  readonly onRankingControlsChange: (controls: OnboardingModelRankingControls) => void
  readonly width: number
  readonly error: string | null
  readonly operation: OnboardingModelChooserOperation | null
  readonly onSelect: (modelId: ProviderModelId) => void
  readonly onExit: () => void
  readonly exitKind: "Skip" | "Close"
}): ReactNode {
  const theme = useTheme()
  const maximumMemoryBytes = Result.isSuccess(hardware)
    ? targetPhysicalMemoryBytes(hardware.value)
    : null
  const selectedRankingScaleIndex = Math.round(
    Math.min(1, Math.max(0, rankingControls.fastToSmart)) * LOCAL_MODEL_RANKING_SCALE_INTERVALS,
  )
  const { selections, ranked, local } = useMemo(() => {
    const ranked = maximumMemoryBytes === null
      ? []
      : rankedLocalModelOptions(options, {
          fastToSmart: rankingControls.fastToSmart,
          memoryBudgetBytes: maximumMemoryBytes,
        }).map((option) => ({ ...option, id: `ranked:${option.id}` }))
    const local = options.filter(({ kind }) => kind === "running" || kind === "stored")
    return { selections: [...ranked, ...local], ranked, local }
  }, [maximumMemoryBytes, options, rankingControls.fastToSmart])
  const [selectedId, setSelectedId] = useState<Option.Option<string>>(Option.none())
  const [cursorIndex, setCursorIndex] = useState(0)
  const [radarTransition, setRadarTransition] = useState<PentagonRadarTransition | null>(null)
  const localScrollRef = useRef<ScrollBoxRenderable | null>(null)
  const downloadScrollRef = useRef<ScrollBoxRenderable | null>(null)
  const activeCursorIndex = Math.min(cursorIndex, Math.max(0, selections.length - 1))
  const activeCursorId = selections[activeCursorIndex]?.id ?? ""
  const operationMatchesSelection = (selection: LocalInferenceSelection): boolean =>
    operation !== null && (operation._tag === "Downloading" || operation._tag === "Configuring"
      ? operation.model.modelId === selection.model.modelId
      : Option.contains(selectionProviderModelId(selection), operation.providerModelId))
  const activeSelectionId = operation === null
    ? Option.none<string>()
    : Option.fromNullable((
        selections.find((selection) =>
          Option.contains(selectedId, selection.id) && operationMatchesSelection(selection))
        ?? selections.find(operationMatchesSelection)
      )?.id)
  const selectedIndex = operation !== null && Option.isNone(activeSelectionId)
    ? -1
    : operation === null
      ? activeCursorIndex
      : selectedInferenceIndex(selections, activeSelectionId)
  const selected = selections[selectedIndex]
  const detailModel = operation?.model ?? selected?.model
  const selectedMemory = detailModel?.servingState._tag === "Assessed"
    && detailModel.servingState.assessment._tag === "Fits"
    ? Option.some(detailModel.servingState.assessment.memory)
    : Option.none<LocalModelMemory>()
  const selectedRadarAxes = detailModel === undefined
    ? Option.none()
    : localModelRadarAxes(detailModel)
  const locked = operation !== null
  const cardWidth = setupBodyWidth(width)
  const wide = isWideSetupLayout(width)
  const leftWidth = wide ? WIDE_LIST_WIDTH : Math.max(1, cardWidth - 6)
  const detailWidth = wide ? Math.max(1, cardWidth - leftWidth - 9) : leftWidth
  const operationWidth = wide ? leftWidth + detailWidth : leftWidth
  const detailContentRows = ONBOARDING_MODEL_DETAIL_ROWS
  const detailPanelRows = detailContentRows + (wide ? 0 : 1)
  const rankedViewportRows = Math.min(RECOMMENDED_VIEWPORT_ROWS, ranked.length)
  const rankedShowOverflow = ranked.length > rankedViewportRows
  const rankedRows = ranked.length > 0
    ? rankedViewportRows
      + MODEL_SECTION_HEADING_ROWS
    : 0
  const sectionGap = local.length > 0 && ranked.length > 0 ? 1 : 0
  const localLayout = onboardingLocalModelLayout({
    wide,
    localCount: local.length,
    detailPanelRows,
    rankedRows,
    sectionGap,
  })
  const localRows = local.length > 0
    ? localLayout.viewportRows
      + MODEL_SECTION_HEADING_ROWS
    : 0
  const listRows = rankedRows + sectionGap + localRows
  const chooserHeight = wide
    ? Math.max(listRows, detailPanelRows)
    : listRows + detailPanelRows
  const choose = useCallback((selection: LocalInferenceSelection) => {
    onSelect(selectionModelId(selection))
  }, [onSelect])

  const moveSelectionTo = useCallback((index: number) => {
    const nextIndex = Math.min(Math.max(0, index), Math.max(0, selections.length - 1))
    const selection = selections[nextIndex]
    if (!selection) return
    const fromAxes = selected === undefined ? Option.none() : localModelRadarAxes(selected.model)
    const toAxes = localModelRadarAxes(selection.model)
    if (selection.id !== selected?.id && Option.isSome(fromAxes) && Option.isSome(toAxes)) {
      setRadarTransition(retargetPentagonRadar(
        pentagonRadarValues(fromAxes.value),
        pentagonRadarValues(toAxes.value),
        radarTransition,
        getAnimationTimeSnapshot(),
      ))
    } else {
      setRadarTransition(null)
    }
    setSelectedId(Option.some(selection.id))
    setCursorIndex(nextIndex)
    const rankedIndex = ranked.indexOf(selection)
    const localIndex = local.indexOf(selection)
    if (rankedIndex >= 0) {
      scrollOnboardingModelPastOverflowIndicators(
        downloadScrollRef.current,
        selection.id,
        rankedIndex,
        ranked.length,
        rankedViewportRows,
      )
    } else if (localIndex >= 0) {
      scrollOnboardingModelPastOverflowIndicators(
        localScrollRef.current,
        selection.id,
        localIndex,
        local.length,
        localLayout.viewportRows,
      )
    }
  }, [local, localLayout.viewportRows, radarTransition, ranked, rankedViewportRows, selected, selections])

  const moveCursorTo = useCallback((index: number) => {
    moveSelectionTo(index)
  }, [moveSelectionTo])

  const adjustControl = useCallback((direction: -1 | 1) => {
    onRankingControlsChange({
      fastToSmart: Math.min(1, Math.max(0,
        rankingControls.fastToSmart + direction / LOCAL_MODEL_RANKING_SCALE_INTERVALS)),
    })
  }, [onRankingControlsChange, rankingControls.fastToSmart])

  useKeyboard(useCallback((key: KeyEvent) => {
    if (locked) return
    if (key.name === "up" || key.name === "k") {
      key.preventDefault()
      moveCursorTo(activeCursorIndex - 1)
      return
    }
    if (key.name === "down" || key.name === "j") {
      key.preventDefault()
      moveCursorTo(activeCursorIndex + 1)
      return
    }
    if (key.name === "left" || key.name === "h") {
      key.preventDefault()
      adjustControl(-1)
      return
    }
    if (key.name === "right" || key.name === "l") {
      key.preventDefault()
      adjustControl(1)
      return
    }
    const cursorSelection = selections[activeCursorIndex]
    if ((key.name === "return" || key.name === "enter") && cursorSelection) {
      key.preventDefault()
      choose(cursorSelection)
      return
    }
    if (key.name === "escape") {
      key.preventDefault()
      onExit()
    }
  }, [activeCursorIndex, adjustControl, choose, locked, moveCursorTo, onExit, selections]))

  const list = (
    <box style={{ width: wide ? leftWidth : "100%", flexDirection: "column", paddingRight: wide ? 1 : 0 }}>
      {ranked.length > 0 && (
        <text style={{ fg: theme.text.supporting }} attributes={TextAttributes.BOLD}>
          RECOMMENDED MODELS
        </text>
      )}
      {ranked.length > 0 && (
        <ModelSectionViewport
          scrollRef={downloadScrollRef}
          rows={rankedViewportRows}
          itemCount={ranked.length}
          showOverflow={rankedShowOverflow}
        >
          {ranked.map((selection, index) => (
            <ModelRow
              key={selection.id}
              selection={selection}
              selected={selection.id === activeCursorId}
              disabled={locked}
              width={leftWidth}
              rowId={onboardingModelRowId(selection.id)}
              rank={index + 1}
              onHover={() => moveSelectionTo(selections.indexOf(selection))}
              onChoose={() => choose(selection)}
            />
          ))}
        </ModelSectionViewport>
      )}
      {local.length > 0 && (
        <text style={{ fg: theme.text.supporting, marginTop: ranked.length > 0 ? 1 : 0 }} attributes={TextAttributes.BOLD}>
          ON THIS COMPUTER
        </text>
      )}
      {local.length > 0 && (
        <ModelSectionViewport
          scrollRef={localScrollRef}
          rows={localLayout.viewportRows}
          itemCount={local.length}
          showOverflow={localLayout.showOverflow}
        >
          {local.map((selection) => (
            <ModelRow
              key={selection.id}
              selection={selection}
              selected={selection.id === activeCursorId}
              disabled={locked}
              width={leftWidth}
              rowId={onboardingModelRowId(selection.id)}
              onHover={() => moveSelectionTo(selections.indexOf(selection))}
              onChoose={() => choose(selection)}
            />
          ))}
        </ModelSectionViewport>
      )}
    </box>
  )

  const contextLabel = selected === undefined
    ? null
    : Option.getOrNull(selectionContextLabel(selected))
  const titleMetadata = contextLabel === null ? null : `${contextLabel} CONTEXT`
  const titleNameWidth = Math.max(
    1,
    detailWidth - (titleMetadata === null ? 0 : titleMetadata.length + 1),
  )
  const memoryWarning = Option.match(selectedMemory, {
    onNone: () => null,
    onSome: ({ currentHeadroomState, systemUseState }) =>
      currentHeadroomState._tag === "Insufficient"
        ? `! Low memory: Free ${formatMemorySize(currentHeadroomState.minimumAdditionalAvailableBytes, { rounding: "up" })} to load`
        : systemUseState._tag === "High"
          ? "! Heavy memory use: Limited memory remains for other apps"
          : null,
  })
  const discoveredLocation = detailModel !== undefined
      && detailModel.catalogMembershipState._tag !== "InCatalog"
    ? discoveredModelLocation(detailModel)
    : null
  const modelSummary = detailModel === undefined
    ? ""
    : detailModel.catalogMembershipState._tag === "InCatalog"
        && detailModel.servingState._tag === "Assessed"
      ? formatModelClassification(
          detailModel.catalogMembershipState.catalogData.parameterization,
          detailModel.servingState.capabilities.vision,
        )
      : discoveredLocation === null
        ? ""
        : `DISCOVERED MODEL · ${discoveredLocation}`
  const modelReleaseRecency = detailModel?.catalogMembershipState._tag === "InCatalog"
    && detailModel.servingState._tag === "Assessed"
    ? formatModelReleaseRecency(detailModel.catalogMembershipState.catalogData.releaseDate)
    : null
  const emptySelectionMessage = "No compatible models found."
  const regularDetails = detailModel ? (
    <>
      <DetailRow width={detailWidth}>
        <text style={{ fg: theme.text.body, flexGrow: 1 }} attributes={TextAttributes.BOLD} wrapMode="none">
          {truncateToDisplayWidth(formatLocalModelDisplayName(detailModel), titleNameWidth)}
        </text>
        {titleMetadata && (
          <text style={{ fg: theme.text.supporting }} wrapMode="none">
            {titleMetadata}
          </text>
        )}
      </DetailRow>
      <box style={{
        width: detailWidth,
        height: MODEL_SUMMARY_ROWS,
        minHeight: MODEL_SUMMARY_ROWS,
        maxHeight: MODEL_SUMMARY_ROWS,
        flexShrink: 0,
        flexDirection: "column",
        overflow: "hidden",
      }}>
        <text style={{ fg: theme.text.supporting, width: detailWidth }} wrapMode="none">
          {modelReleaseRecency !== null && (
            <span fg={theme.text.detail}>{`${modelReleaseRecency} · `}</span>
          )}
          {truncateToDisplayWidth(
            modelSummary,
            Math.max(0, detailWidth - (modelReleaseRecency === null ? 0 : `${modelReleaseRecency} · `.length)),
          )}
        </text>
      </box>
      <box style={{ height: 1, flexShrink: 0 }} />
      {Option.match(selectedRadarAxes, {
        onNone: () => <box style={{ height: PENTAGON_RADAR_ROWS, minHeight: PENTAGON_RADAR_ROWS, flexShrink: 0 }} />,
        onSome: (axes) => (
          <PentagonRadarView
            axes={axes}
            transition={radarTransition}
            columns={Math.min(PENTAGON_RADAR_COLUMNS, detailWidth)}
          />
        ),
      })}
      {operation === null && memoryWarning && (
        <text style={{ fg: theme.status.warning, width: detailWidth }} wrapMode="none">{memoryWarning}</text>
      )}
    </>
  ) : (
    <text style={{ fg: theme.text.supporting }}>{emptySelectionMessage}</text>
  )
  const details = (
    <box style={{
      flexDirection: "column",
      flexGrow: wide ? 1 : 0,
      minWidth: 0,
      height: detailPanelRows,
      minHeight: detailPanelRows,
      maxHeight: detailPanelRows,
      overflow: "hidden",
      paddingLeft: wide ? 2 : 0,
      borderStyle: "single",
      border: wide ? ["left"] : ["top"],
      borderColor: theme.border.standard,
      customBorderChars: BOX_CHARS,
    }}>
      {regularDetails}
    </box>
  )
  const enterAction = onboardingSelectionEnterAction(selected?.kind)
  const exitHint = exitKind === "Close" ? "close setup" : "skip for now"
  const selectionHint = enterAction === null
    ? `Esc ${exitHint}`
    : `↑/↓ choose · Enter to ${enterAction} · Esc ${exitHint}`
  const interactionHint = operation?._tag === "Downloading"
      ? operation.starting
        ? "Starting download…"
        : "Download in progress · Esc cancel"
      : operation?._tag === "Configuring"
        ? "Configuring model…"
      : operation?._tag === "Activating"
        ? operation.status._tag === "Failed"
          ? "Model loading failed"
          : operation.status._tag === "Cancelling"
            ? "Cancelling loading…"
            : operation.status._tag === "Stopping"
            ? "Stopping model…"
            : operation.status._tag === "Preparing" || operation.status._tag === "Loading"
              ? "Loading model into memory…"
              : "Finishing setup…"
    : Option.exists(
        selectedMemory,
        ({ currentHeadroomState }) => currentHeadroomState._tag === "Insufficient",
      )
      ? selected?.kind === "stored"
        ? "Close memory-intensive apps, then Enter to load · Esc choose another"
        : selectionHint
      : selectionHint

  return (
    <OnboardingSetupCard
      width={width}
      stage={operation === null ? "choose" : "install"}
      hardware={hardware}
      unexpectedError={error}
      footer={(
        <>
          <text style={{ fg: theme.text.guidance }} wrapMode="none">You can switch models or download more anytime from /settings.</text>
          <text style={{ fg: theme.text.supporting }} wrapMode="none">{interactionHint}</text>
        </>
      )}
    >
      {operation === null
        ? (
            <box style={{
              flexDirection: "column",
              width: "100%",
              height: ONBOARDING_RANKING_CONTROL_ROWS,
              minHeight: ONBOARDING_RANKING_CONTROL_ROWS,
              maxHeight: ONBOARDING_RANKING_CONTROL_ROWS,
              flexShrink: 0,
            }}>
              <text selectable={false} style={{ fg: theme.text.body }} wrapMode="none">
                {" ".repeat(FAST_TO_SMART_TRACK_LEFT_PADDING)}
                {LOCAL_MODEL_RANKING_SCALE_LABELS.map((_, index) => (
                  <Fragment key={index}>
                    <span fg={index === selectedRankingScaleIndex ? theme.accent : theme.text.body}>
                      {index === 0 ? "├" : index === LOCAL_MODEL_RANKING_SCALE_INTERVALS ? "┤" : "┼"}
                    </span>
                    {index < LOCAL_MODEL_RANKING_SCALE_INTERVALS
                      ? "─".repeat(FAST_TO_SMART_SEGMENT_COLUMNS - 1)
                      : ""}
                  </Fragment>
                ))}
                {wide && (
                  <>
                    {"    "}
                    <span fg={theme.text.disabled}>←/→ change preference</span>
                  </>
                )}
              </text>
              <text selectable={false} style={{ fg: theme.text.body }} wrapMode="none">
                {FAST_TO_SMART_LABEL_LAYOUT.map(({ label, leadingSpaces }, index) => (
                  <Fragment key={label}>
                    {" ".repeat(leadingSpaces)}
                    <span fg={index === selectedRankingScaleIndex ? theme.accent : theme.text.body}>{label}</span>
                  </Fragment>
                ))}
              </text>
              {wide ? (
                <box style={{ height: 1, minHeight: 1, maxHeight: 1, flexShrink: 0 }} />
              ) : (
                <text selectable={false} style={{ fg: theme.text.disabled }} wrapMode="none">
                  {" ".repeat(FAST_TO_SMART_TRACK_LEFT_PADDING)}←/→ change preference
                </text>
              )}
            </box>
          )
        : operation._tag === "Downloading"
          ? (
              <OnboardingModelDownloadProgress
                model={operation.model}
                width={operationWidth}
                operation={operation}
              />
            )
          : operation._tag === "Activating"
            ? (
                <OnboardingModelLoadProgress
                  model={operation.model}
                  status={operation.status}
                  width={operationWidth}
                  onCancel={operation.onCancel}
                  onRetry={operation.onRetry}
                  onChooseAnother={operation.onChooseAnother}
                />
              )
            : <OnboardingModelConfiguringProgress width={operationWidth} />}
      <box style={{
        flexDirection: wide ? "row" : "column",
        width: "100%",
        height: chooserHeight,
        minHeight: chooserHeight,
        maxHeight: chooserHeight,
        overflow: "hidden",
      }}>
        {list}
        {details}
      </box>
    </OnboardingSetupCard>
  )
}

export function OnboardingModelPreparation({
  hardware,
  progress,
  error,
  width,
  onExit,
  exitKind,
}: {
  readonly hardware: LocalInferenceHardwareResult
  readonly progress: readonly LocalModelDiscoveryProgressStep[]
  readonly error: string | null
  readonly width: number
  readonly onExit: (() => void) | undefined
  readonly exitKind: "Skip" | "Close" | null
}): ReactNode {
  const theme = useTheme()
  const lines = localInferenceProgressLines(progress)
    .filter(({ id }) => id !== "hardware")
  const spinner = useSpinnerFrame(
    Result.isInitial(hardware)
      || lines.some(({ state }) => state === "running"),
  )
  useKeyboard(useCallback((key: KeyEvent) => {
    if (key.name === "escape" && onExit !== undefined) {
      key.preventDefault()
      onExit()
    }
  }, [onExit]))
  return (
    <OnboardingSetupCard
      width={width}
      stage="choose"
      title="Preparing local models"
      hardware={hardware}
      spinnerFrame={spinner}
      footer={exitKind === null ? undefined : (
        <text style={{ fg: theme.text.supporting }}>
          {exitKind === "Close" ? "Esc close setup" : "Esc skip for now"}
        </text>
      )}
    >
      {lines.map((line) => (
        <text key={line.id} style={{ fg: line.state === "pending" ? theme.text.supporting : theme.text.body }}>
          <span fg={line.state === "completed" ? theme.status.success : line.state === "failed" ? theme.status.failure : line.state === "running" ? theme.accent : theme.text.supporting}>
            {line.state === "completed" ? "✓ " : line.state === "failed" ? "! " : line.state === "running" ? `${spinner} ` : "○ "}
          </span>
          {line.label}<span fg={line.state === "failed" ? theme.status.failure : theme.text.supporting}>{line.metadata}</span>
        </text>
      ))}
      {error && <text style={{ fg: theme.status.failure }}>{error}</text>}
    </OnboardingSetupCard>
  )
}

export function OnboardingModelExiting({
  hardware,
  width,
}: {
  readonly hardware: LocalInferenceHardwareResult
  readonly width: number
}): ReactNode {
  const theme = useTheme()
  const spinner = useSpinnerFrame(true)
  return (
    <OnboardingSetupCard
      width={width}
      stage="choose"
      title="Finishing onboarding"
      hardware={hardware}
      spinnerFrame={spinner}
      footer={<text style={{ fg: theme.text.supporting }}>Setup will close when this finishes.</text>}
    >
      <text style={{ fg: theme.text.body }}>{spinner} Saving onboarding completion…</text>
    </OnboardingSetupCard>
  )
}
