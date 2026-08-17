import { useCallback, useMemo, useRef, useState, type ReactNode, type Ref } from "react"
import { TextAttributes, type KeyEvent, type ScrollBoxRenderable } from "@opentui/core"
import { useKeyboard } from "@opentui/react"
import { Result } from "@effect-atom/atom-react"
import { Option } from "effect"
import {
  getAnimationTimeSnapshot,
  clampTextToVisualLines,
  truncateToDisplayWidth,
  formatLocalModelDisplayName,
  localModelConfigurationId,
  type LocalModelOption,
  type LocalInferenceHardwareResult,
} from "@magnitudedev/client-common"
import type {
  LocalModel,
  LocalModelMemory,
  LocalModelRecommendationProgressStep,
  ModelInstanceFailure,
  ModelServingConfigurationId,
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
import {
  describeLocalHardwareSummary,
  formatBytes,
  formatDownloadBytes,
  localInferenceProgressLines,
  selectedInferenceIndex,
  selectionContextLabel,
  selectionConfigurationId,
  selectionProviderModelId,
  type LocalInferenceSelection,
} from "../local-inference/view-model"
import { localModelRadarAxes } from "@magnitudedev/client-common"
import { formatModelClassification } from "../local-inference/model-classification"
import { discoveredModelLocation } from "./discovered-model"
import {
  OnboardingModelDownloadProgress,
  ONBOARDING_MODEL_DOWNLOAD_ROWS,
} from "./model-status"

const SECTION_VIEWPORT_ROWS = 4
const MODEL_TITLE_ROWS = 1
const MODEL_SUMMARY_ROWS = 1
const MODEL_SUMMARY_RADAR_GAP_ROWS = 1
const RECOMMENDATION_HEADING_ROWS = 1
const RECOMMENDATION_ROWS = 3
const WIDE_LIST_WIDTH = 38
const WIDE_CHOOSER_MIN_WIDTH = 105

const onboardingModelRowId = (selectionId: string): string =>
  `onboarding-model:${selectionId}`

export const scrollOnboardingModelIntoView = (
  scrollbox: Pick<ScrollBoxRenderable, "scrollChildIntoView"> | null,
  selectionId: string,
): void => {
  scrollbox?.scrollChildIntoView(onboardingModelRowId(selectionId))
}

const setupCardWidth = (width: number): number => Math.max(1, Math.min(110, width - 2))

const intentLabel = (intent: "balanced" | "smartest" | "fastest" | "lightweight"): string => {
  if (intent === "smartest") return "Smartest"
  if (intent === "fastest") return "Fastest"
  if (intent === "lightweight") return "Lightweight"
  return "Balanced"
}

export const onboardingModelActionLabel = (selection: LocalInferenceSelection): string => {
  if (selection.kind === "running") return "Loaded"
  if (selection.recommendations.length > 0) {
    return selection.recommendations.map(({ intent }) => intentLabel(intent)).join(" / ")
  }
  if (selection.kind === "recommendation") return "Download"
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
  onHover,
  onChoose,
}: {
  readonly selection: LocalInferenceSelection
  readonly selected: boolean
  readonly disabled: boolean
  readonly width: number
  readonly rowId: string
  readonly onHover: () => void
  readonly onChoose: () => void
}): ReactNode => {
  const theme = useTheme()
  const action = onboardingModelActionLabel(selection)
  const enabled = selection.kind !== "recommendation"
    || selection.recommendations.length > 0
  const markerWidth = 2
  const gap = 2
  const nameWidth = Math.max(1, width - markerWidth - gap - action.length - 1)
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
        {selected ? "› " : "  "}{truncateToDisplayWidth(onboardingModelRowName(selection), nameWidth).padEnd(nameWidth)}
        {"  "}
        <span fg={selection.kind === "running"
          ? theme.status.success
          : selection.recommendations.length > 0 || selected
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
  children,
}: {
  readonly scrollRef: Ref<ScrollBoxRenderable | null>
  readonly rows: number
  readonly children: ReactNode
}): ReactNode => (
  <scrollbox
    ref={scrollRef}
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
)

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

const minimumBytesLabel = (bytes: number): string => {
  const gib = bytes / 1024 ** 3
  const precision = gib >= 10 ? 10 : 100
  return `${(Math.ceil(gib * precision) / precision).toFixed(gib >= 10 ? 1 : 2)} GiB`
}

const compactMemoryLabel = (bytes: number): string =>
  `${Math.max(0.1, bytes / 1024 ** 3).toFixed(1)} GB`

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
    <text key={`${row.name}:${row.details.join(":")}`} style={{ width }} wrapMode="word">
      <span fg={theme.text.detail}>{row.name}</span>
      <span fg={theme.text.supporting}>{` · ${row.details.join(" · ")}`}</span>
    </text>
  ))
  if (Result.isFailure(hardware)) {
    return <text style={{ fg: theme.status.failure, width }}>! Hardware detection failed</text>
  }
  return (
    <text style={{ width }}>
      <span fg={theme.accent}>{spinnerFrame} </span>
      <span fg={theme.text.detail}>Detecting hardware…</span>
    </text>
  )
}

const OnboardingSetupCard = ({
  cardWidth,
  title,
  hardware,
  spinnerFrame = spinnerFrameAt(0),
  children,
}: {
  readonly cardWidth: number
  readonly title: string
  readonly hardware: LocalInferenceHardwareResult
  readonly spinnerFrame?: string
  readonly children: ReactNode
}): ReactNode => {
  const theme = useTheme()
  return (
    <box style={{ width: "100%", flexGrow: 1, alignItems: "center", justifyContent: "center" }}>
      <box style={{
        width: cardWidth,
        borderStyle: "single",
        borderColor: theme.border.standard,
        customBorderChars: BOX_CHARS,
        paddingLeft: 2,
        paddingRight: 2,
        flexDirection: "column",
      }}>
        <text style={{ fg: theme.text.body }} attributes={TextAttributes.BOLD}>{title}</text>
        <OnboardingHardwareContext
          hardware={hardware}
          width={Math.max(1, cardWidth - 6)}
          spinnerFrame={spinnerFrame}
        />
        <box style={{ height: 1 }} />
        {children}
      </box>
    </box>
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
      readonly phase: "Loading" | "Stopping" | "Ready" | "Failed"
      readonly failure: ModelInstanceFailure | null
      readonly onRetry: () => void
      readonly onChooseAnother: () => void
    }

export const onboardingSelectionEnterAction = (
  kind: LocalInferenceSelection["kind"] | undefined,
): "download" | "load" | "select" | null => {
  if (kind === "recommendation") return "download"
  if (kind === "stored") return "load"
  if (kind === "running") return "select"
  return null
}

export const onboardingModelDetailRows = ({
  recommendation,
  memoryWarning,
  downloadRows,
  modelSummaryRadarGap,
}: {
  readonly recommendation: boolean
  readonly memoryWarning: boolean
  readonly downloadRows: number
  readonly modelSummaryRadarGap: boolean
}): number => MODEL_TITLE_ROWS
  + MODEL_SUMMARY_ROWS
  + (modelSummaryRadarGap ? MODEL_SUMMARY_RADAR_GAP_ROWS : 0)
  + PENTAGON_RADAR_ROWS
  + (recommendation
    ? RECOMMENDATION_HEADING_ROWS + RECOMMENDATION_ROWS
    : (memoryWarning ? 1 : 0) + downloadRows)

const ONBOARDING_IDLE_MODEL_DETAIL_ROWS = onboardingModelDetailRows({
  recommendation: true,
  memoryWarning: false,
  downloadRows: 0,
  modelSummaryRadarGap: true,
})

const ONBOARDING_DOWNLOADING_DETAIL_ROWS = onboardingModelDetailRows({
  recommendation: false,
  memoryWarning: false,
  downloadRows: ONBOARDING_MODEL_DOWNLOAD_ROWS,
  modelSummaryRadarGap: true,
})

export const ONBOARDING_MODEL_DETAIL_ROWS = Math.max(
  ONBOARDING_IDLE_MODEL_DETAIL_ROWS,
  ONBOARDING_DOWNLOADING_DETAIL_ROWS,
)

export const onboardingLocalModelViewportRows = ({
  wide,
  localCount,
  detailPanelRows,
  downloadRows,
  sectionGap,
}: {
  readonly wide: boolean
  readonly localCount: number
  readonly detailPanelRows: number
  readonly downloadRows: number
  readonly sectionGap: number
}): number => {
  if (localCount === 0) return 0
  if (!wide) return Math.min(SECTION_VIEWPORT_ROWS, localCount)
  const localHeadingRows = 1
  return Math.max(1, detailPanelRows - downloadRows - sectionGap - localHeadingRows)
}

export function OnboardingModelChooser({
  hardware,
  options,
  width,
  error,
  operation,
  onSelect,
  onExit,
  exitKind,
}: {
  readonly hardware: LocalInferenceHardwareResult
  readonly options: readonly LocalModelOption[]
  readonly width: number
  readonly error: string | null
  readonly operation: OnboardingModelChooserOperation | null
  readonly onSelect: (configurationId: ModelServingConfigurationId) => void
  readonly onExit: () => void
  readonly exitKind: "Skip" | "Close"
}): ReactNode {
  const theme = useTheme()
  const { selections, downloads, local } = useMemo(() => {
    const eligible = options.filter((selection) =>
      selection.kind !== "recommendation"
        || selection.recommendations.length > 0)
    const downloads = eligible.filter(({ kind }) => kind === "recommendation")
    const local = eligible.filter(({ kind }) => kind === "running" || kind === "stored")
    return { selections: [...downloads, ...local], downloads, local }
  }, [options])
  const [selectedId, setSelectedId] = useState<Option.Option<string>>(Option.none())
  const [radarTransition, setRadarTransition] = useState<PentagonRadarTransition | null>(null)
  const localScrollRef = useRef<ScrollBoxRenderable | null>(null)
  const downloadScrollRef = useRef<ScrollBoxRenderable | null>(null)
  const activeSelectionId = operation === null
    ? Option.none<string>()
    : Option.fromNullable(selections.find((selection) =>
      operation._tag === "Downloading" || operation._tag === "Configuring"
      ? Option.exists(localModelConfigurationId(operation.model), (configurationId) =>
          Option.contains(selectionConfigurationId(selection), configurationId))
      : Option.contains(selectionProviderModelId(selection), operation.providerModelId))?.id)
  const selectedIndex = operation !== null && Option.isNone(activeSelectionId)
    ? -1
    : selectedInferenceIndex(
        selections,
        Option.isSome(activeSelectionId) ? activeSelectionId : selectedId,
      )
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
  const cardWidth = setupCardWidth(width)
  const wide = cardWidth >= WIDE_CHOOSER_MIN_WIDTH
  const leftWidth = wide ? WIDE_LIST_WIDTH : Math.max(1, cardWidth - 6)
  const detailWidth = wide ? Math.max(1, cardWidth - leftWidth - 9) : leftWidth
  const downloadOperation = operation?._tag === "Downloading" ? operation : null
  const detailContentRows = ONBOARDING_MODEL_DETAIL_ROWS
  const detailPanelRows = detailContentRows + (wide ? 0 : 1)
  const downloadViewportRows = Math.min(SECTION_VIEWPORT_ROWS, downloads.length)
  const downloadRows = downloads.length > 0 ? downloadViewportRows + 1 : 0
  const sectionGap = local.length > 0 && downloads.length > 0 ? 1 : 0
  const localViewportRows = onboardingLocalModelViewportRows({
    wide,
    localCount: local.length,
    detailPanelRows,
    downloadRows,
    sectionGap,
  })
  const localRows = local.length > 0 ? localViewportRows + 1 : 0
  const listRows = downloadRows + sectionGap + localRows
  const chooserHeight = wide
    ? Math.max(listRows, detailPanelRows)
    : listRows + detailPanelRows
  const choose = useCallback((selection: LocalInferenceSelection) => {
    const configurationId = selectionConfigurationId(selection)
    if (Option.isSome(configurationId)) onSelect(configurationId.value)
  }, [onSelect])

  const moveSelectionTo = useCallback((index: number) => {
    const selection = selections[index]
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
    scrollOnboardingModelIntoView(
      selection.kind === "recommendation" ? downloadScrollRef.current : localScrollRef.current,
      selection.id,
    )
  }, [radarTransition, selected, selections])

  useKeyboard(useCallback((key: KeyEvent) => {
    if (locked) return
    if (key.name === "up" || key.name === "k") {
      key.preventDefault()
      moveSelectionTo(Math.max(0, selectedIndex - 1))
      return
    }
    if (key.name === "down" || key.name === "j" || key.name === "tab") {
      key.preventDefault()
      moveSelectionTo(Math.min(
        Math.max(0, selections.length - 1),
        selectedIndex + 1,
      ))
      return
    }
    if ((key.name === "return" || key.name === "enter") && selected) {
      key.preventDefault()
      choose(selected)
      return
    }
    if (key.name === "escape") {
      key.preventDefault()
      onExit()
    }
  }, [choose, locked, moveSelectionTo, onExit, selected, selectedIndex, selections.length]))

  const list = (
    <box style={{ width: wide ? leftWidth : "100%", flexDirection: "column", paddingRight: wide ? 1 : 0 }}>
      {downloads.length > 0 && (
        <text style={{ fg: theme.text.supporting }} attributes={TextAttributes.BOLD}>
          AVAILABLE TO DOWNLOAD
        </text>
      )}
      {downloads.length > 0 && (
        <ModelSectionViewport scrollRef={downloadScrollRef} rows={downloadViewportRows}>
          {downloads.map((selection) => (
            <ModelRow
              key={selection.id}
              selection={selection}
              selected={selection.id === selected?.id}
              disabled={locked}
              width={leftWidth}
              rowId={onboardingModelRowId(selection.id)}
              onHover={() => moveSelectionTo(selections.indexOf(selection))}
              onChoose={() => choose(selection)}
            />
          ))}
        </ModelSectionViewport>
      )}
      {local.length > 0 && (
        <text style={{ fg: theme.text.supporting, marginTop: downloads.length > 0 ? 1 : 0 }} attributes={TextAttributes.BOLD}>
          ON THIS COMPUTER
        </text>
      )}
      {local.length > 0 && (
        <ModelSectionViewport scrollRef={localScrollRef} rows={localViewportRows}>
          {local.map((selection) => (
            <ModelRow
              key={selection.id}
              selection={selection}
              selected={selection.id === selected?.id}
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
        ? `! Low memory: Free ${compactMemoryLabel(currentHeadroomState.minimumAdditionalAvailableBytes)} to load`
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
  const recommendationBodyRows = Math.max(
    1,
    RECOMMENDATION_ROWS - (memoryWarning === null ? 0 : 1),
  )
  const recommendationExplanation = selected?.recommendations[0] !== undefined
    ? clampTextToVisualLines(
        selected.recommendations[0].explanation,
        detailWidth,
        recommendationBodyRows,
      )
    : ""
  const showRecommendationExplanation = selected !== undefined
    && selected.recommendations.length > 0
    && operation?._tag !== "Downloading"
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
          {truncateToDisplayWidth(modelSummary, detailWidth)}
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
      {showRecommendationExplanation && (
        <box style={{
          height: RECOMMENDATION_HEADING_ROWS + RECOMMENDATION_ROWS,
          minHeight: RECOMMENDATION_HEADING_ROWS + RECOMMENDATION_ROWS,
          maxHeight: RECOMMENDATION_HEADING_ROWS + RECOMMENDATION_ROWS,
          flexDirection: "column",
          flexShrink: 0,
          overflow: "hidden",
        }}>
            <text style={{ fg: theme.text.supporting, width: detailWidth }} attributes={TextAttributes.BOLD} wrapMode="none">
              WHY THIS MODEL
            </text>
            <box style={{
              height: recommendationBodyRows,
              minHeight: recommendationBodyRows,
              maxHeight: recommendationBodyRows,
              flexDirection: "column",
              flexShrink: 0,
              overflow: "hidden",
            }}>
              <text style={{ fg: theme.text.supporting, width: detailWidth }} wrapMode="none">
                {recommendationExplanation}
              </text>
            </box>
            {memoryWarning && (
              <text style={{ fg: theme.status.warning, width: detailWidth }} wrapMode="none">{memoryWarning}</text>
            )}
        </box>
      )}
      {!showRecommendationExplanation && downloadOperation === null && memoryWarning && (
        <text style={{ fg: theme.status.warning, width: detailWidth }} wrapMode="none">{memoryWarning}</text>
      )}
      <box style={{ flexGrow: 1 }} />
      {downloadOperation !== null && (
        <OnboardingModelDownloadProgress
          model={downloadOperation.model}
          width={detailWidth}
          operation={downloadOperation}
        />
      )}
    </>
  ) : (
    <text style={{ fg: theme.text.supporting }}>{emptySelectionMessage}</text>
  )
  const detailsContent = operation?._tag === "Activating" ? (
    <OnboardingModelLoadingDetails
      displayName={formatLocalModelDisplayName(operation.model)}
      width={detailWidth}
      height={detailContentRows}
      phase={operation.phase}
      failed={operation.failure}
      onRetry={operation.onRetry}
      onChooseAnother={operation.onChooseAnother}
    />
  ) : regularDetails
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
      {detailsContent}
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
        ? operation.phase === "Failed"
          ? "Model loading failed"
          : operation.phase === "Stopping"
            ? "Stopping model…"
            : operation.phase === "Loading"
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
      cardWidth={cardWidth}
      title="Choose a local model"
      hardware={hardware}
    >
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
      {error && <text style={{ fg: theme.status.failure, marginTop: 1 }} wrapMode="none">{error}</text>}
      <box style={{ height: 1 }} />
      <text style={{ fg: theme.text.guidance }} wrapMode="none">You can switch models or download more anytime from /settings.</text>
      <text style={{ fg: theme.text.supporting }} wrapMode="none">{interactionHint}</text>
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
  readonly progress: readonly LocalModelRecommendationProgressStep[]
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
  const cardWidth = setupCardWidth(width)
  useKeyboard(useCallback((key: KeyEvent) => {
    if (key.name === "escape" && onExit !== undefined) {
      key.preventDefault()
      onExit()
    }
  }, [onExit]))
  return (
    <OnboardingSetupCard
      cardWidth={cardWidth}
      title="Preparing local models"
      hardware={hardware}
      spinnerFrame={spinner}
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
      <box style={{ height: 1 }} />
      {exitKind !== null && (
        <text style={{ fg: theme.text.supporting }}>
          {exitKind === "Close" ? "Esc close setup" : "Esc skip for now"}
        </text>
      )}
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
      cardWidth={setupCardWidth(width)}
      title="Finishing onboarding"
      hardware={hardware}
      spinnerFrame={spinner}
    >
      <text style={{ fg: theme.text.body }}>{spinner} Saving onboarding completion…</text>
      <box style={{ height: 1 }} />
      <text style={{ fg: theme.text.supporting }}>Setup will close when this finishes.</text>
    </OnboardingSetupCard>
  )
}

function OnboardingModelLoadingDetails({
  displayName,
  width,
  height,
  phase,
  failed,
  onRetry,
  onChooseAnother,
}: {
  readonly displayName: string
  readonly width: number
  readonly height: number
  readonly phase: "Loading" | "Stopping" | "Ready" | "Failed"
  readonly failed: ModelInstanceFailure | null
  readonly onRetry: () => void
  readonly onChooseAnother: () => void
}): ReactNode {
  const theme = useTheme()
  const [hovered, setHovered] = useState<"retry" | "choose" | null>(null)
  const spinner = useSpinnerFrame(failed === null)
  return (
    <box style={{
      width,
      height,
      minHeight: height,
      maxHeight: height,
      flexShrink: 0,
      flexDirection: "column",
      overflow: "hidden",
    }}>
      <DetailRow width={width}>
        <text style={{ fg: theme.text.body, width }} attributes={TextAttributes.BOLD} wrapMode="none">
          {truncateToDisplayWidth(
            failed
              ? `Couldn’t load ${displayName}`
              : phase === "Stopping"
                ? `Stopping ${displayName}`
                : phase === "Loading"
                  ? `Loading ${displayName} into memory`
                  : `Finishing setup for ${displayName}`,
            width,
          )}
        </text>
      </DetailRow>
      <box style={{ height: 1 }} />
      {failed ? (
        <>
          {"_tag" in failed && failed._tag === "LowMemory" ? (
            <box style={{ width, flexShrink: 0, flexDirection: "column" }}>
              <text style={{ fg: theme.status.warning, width }} attributes={TextAttributes.BOLD}>
                ! Not enough memory available
              </text>
              <box style={{ height: 1 }} />
              <text style={{ fg: theme.text.body, width }} wrapMode="word">
                {`Free at least ${minimumBytesLabel(failed.minimumAdditionalAvailableBytes)} and try again.`}
              </text>
              <text style={{ fg: theme.text.supporting, width }} wrapMode="word">
                Close memory-intensive applications or choose a smaller model.
              </text>
              <box style={{ height: 1 }} />
              <text style={{ fg: theme.text.supporting, width }}>
                {`Needed at attempt    ${formatBytes(failed.loadBoundaryBytes)}`}
              </text>
              <text style={{ fg: theme.text.supporting, width }}>
                {`Available at attempt ${formatBytes(failed.allocationHeadroomBytes)}`}
              </text>
              <box style={{ height: 1 }} />
            </box>
          ) : (
            <box style={{ width, height: 5, flexShrink: 0, flexDirection: "column", overflow: "hidden" }}>
              <text style={{ fg: theme.status.failure, width }} wrapMode="word">{failed.message}</text>
            </box>
          )}
          <box style={{ flexDirection: "row", gap: 2 }}>
            <Button onClick={onRetry} onMouseOver={() => setHovered("retry")} onMouseOut={() => setHovered(null)}>
              <text style={{ fg: hovered === "retry" ? theme.accent : theme.text.body }}>Retry loading</text>
            </Button>
            <Button onClick={onChooseAnother} onMouseOver={() => setHovered("choose")} onMouseOut={() => setHovered(null)}>
              <text style={{ fg: hovered === "choose" ? theme.accent : theme.text.body }}>Choose another model</text>
            </Button>
          </box>
        </>
      ) : (
        <box style={{ width, flexDirection: "row" }}>
          <text style={{ fg: theme.accent, width: 2, flexShrink: 0 }} wrapMode="none">
            {spinner}
          </text>
          <text style={{ fg: theme.text.supporting, width: Math.max(1, width - 2) }} wrapMode="none">
            {phase === "Loading"
              ? "Loading model weights…"
              : phase === "Stopping"
                ? "Stopping model…"
                : "Finishing setup…"}
          </text>
        </box>
      )}
    </box>
  )
}
