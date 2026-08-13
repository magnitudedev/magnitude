import { useCallback, useMemo, useRef, useState, type ReactNode, type Ref } from "react"
import { TextAttributes, type KeyEvent, type ScrollBoxRenderable } from "@opentui/core"
import { useKeyboard } from "@opentui/react"
import { Result } from "@effect-atom/atom-react"
import { Option } from "effect"
import {
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
import { spinnerFrameAt, useSpinnerFrame } from "../../hooks/use-spinner-frame"
import { useTheme } from "../../hooks/use-theme"
import { BOX_CHARS } from "../../utils/ui-constants"
import {
  describeLocalHardwareSummary,
  formatBytes,
  localInferenceProgressLines,
  performanceRangeSpeedLabel,
  selectedInferenceIndex,
  selectionConfigurationId,
  selectionMetadata,
  selectionProviderModelId,
  type LocalInferenceSelection,
} from "../local-inference/view-model"
import { slate } from "../../utils/theme"
import { OnboardingModelDownloadDetails } from "./download-details"

const SECTION_VIEWPORT_ROWS = 4
const DESCRIPTION_ROWS = 5
const DETAIL_BASE_ROWS = 9
const WIDE_LIST_WIDTH = 42

const onboardingModelRowId = (selectionId: string): string =>
  `onboarding-model:${selectionId}`

export const scrollOnboardingModelIntoView = (
  scrollbox: Pick<ScrollBoxRenderable, "scrollChildIntoView"> | null,
  selectionId: string,
): void => {
  scrollbox?.scrollChildIntoView(onboardingModelRowId(selectionId))
}

const setupCardWidth = (width: number): number => Math.max(1, Math.min(96, width - 2))

const intentLabel = (intent: "balanced" | "best_quality" | "fastest" | "lightweight"): string => {
  if (intent === "best_quality") return "Best Quality"
  if (intent === "fastest") return "Fastest"
  if (intent === "lightweight") return "Lightweight"
  return "Balanced"
}

const actionLabel = (selection: LocalInferenceSelection): string => {
  if (selection.kind === "running") return "Loaded"
  if (selection.kind === "recommendation") {
    return Option.isSome(selection.recommendation)
      ? intentLabel(selection.recommendation.value.intent)
      : "Download"
  }
  return "Load"
}

const onboardingSelection = (
  selection: LocalInferenceSelection,
): ProviderModelId | null => Option.getOrNull(selectionProviderModelId(selection))

export const onboardingModelRowName = (
  selection: LocalInferenceSelection,
): string => selection.kind === "recommendation"
  ? selection.model.presentation.displayName
  : formatLocalModelDisplayName(selection.model)

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
  const action = actionLabel(selection)
  const enabled = selection.kind !== "recommendation"
    || Option.isSome(selection.recommendation)
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
        style={{ fg: selected ? theme.primary : enabled ? theme.foreground : theme.muted }}
        attributes={selected ? TextAttributes.BOLD : TextAttributes.NONE}
        wrapMode="none"
      >
        {selected ? "› " : "  "}{truncateToDisplayWidth(onboardingModelRowName(selection), nameWidth).padEnd(nameWidth)}
        {"  "}
        <span fg={selection.kind === "running"
          ? theme.success
          : selection.kind === "recommendation" || selected
            ? theme.primary
            : theme.muted}>
          {action}
        </span>
      </text>
    </Button>
  )
}

const ModelSectionViewport = ({
  scrollRef,
  children,
}: {
  readonly scrollRef: Ref<ScrollBoxRenderable | null>
  readonly children: ReactNode
}): ReactNode => (
  <scrollbox
    ref={scrollRef}
    scrollX={false}
    scrollbarOptions={{ visible: false }}
    style={{
      flexShrink: 0,
      rootOptions: {
        height: SECTION_VIEWPORT_ROWS,
        minHeight: SECTION_VIEWPORT_ROWS,
        maxHeight: SECTION_VIEWPORT_ROWS,
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

const memoryGuidanceRows = (memory: LocalModelMemory): number =>
  memory.currentHeadroomState._tag === "Insufficient" ? 7 : 3

const ModelMemoryGuidanceDetails = ({
  memory,
  systemUsedBytes,
  width,
}: {
  readonly memory: LocalModelMemory
  readonly systemUsedBytes: number | null
  readonly width: number
}): ReactNode => {
  const theme = useTheme()
  const current = memory.currentHeadroomState
  if (current._tag === "Insufficient") {
    return (
      <box style={{ width, flexDirection: "column", flexShrink: 0 }}>
        <text style={{ fg: theme.muted, width }} attributes={TextAttributes.BOLD}>MEMORY</text>
        <text style={{ fg: theme.warning, width }} wrapMode="none">
          {`! Low memory: Free ${compactMemoryLabel(current.minimumAdditionalAvailableBytes)} to load`}
        </text>
        <text style={{ fg: theme.muted, width }} wrapMode="word">
          {systemUsedBytes === null
            ? "This model fits on total memory but system memory is currently heavily used."
            : `This model fits on total memory but system is currently using ${compactMemoryLabel(systemUsedBytes)}.`}
        </text>
        <text style={{ fg: theme.muted, width }} wrapMode="word">
          Close other memory-intensive apps to run this model.
        </text>
      </box>
    )
  }
  const guidance = memory.systemUseState._tag === "High"
      ? {
          color: theme.warning,
          text: "! Tight fit: Limited memory remains for other apps",
        }
      : {
          color: theme.muted,
          text: `Model allocation ${formatBytes(memory.totalRequiredBytes)}`,
        }
  return (
    <box style={{ width, flexDirection: "column", flexShrink: 0 }}>
      <text style={{ fg: theme.muted, width }} attributes={TextAttributes.BOLD}>MEMORY</text>
      <text style={{ fg: guidance.color, width }} wrapMode="none">{guidance.text}</text>
    </box>
  )
}

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
      <span fg={slate[300]}>{row.name}</span>
      <span fg={slate[400]}>{` · ${row.details.join(" · ")}`}</span>
    </text>
  ))
  if (Result.isFailure(hardware)) {
    return <text style={{ fg: theme.error, width }}>! Hardware detection failed</text>
  }
  return (
    <text style={{ width }}>
      <span fg={theme.primary}>{spinnerFrame} </span>
      <span fg={slate[300]}>Detecting hardware…</span>
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
        borderColor: theme.border,
        customBorderChars: BOX_CHARS,
        paddingTop: 1,
        paddingBottom: 1,
        paddingLeft: 2,
        paddingRight: 2,
        flexDirection: "column",
      }}>
        <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>{title}</text>
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
      readonly cancelError: string | null
      readonly onCancel: () => void
    }
  | {
      readonly _tag: "Configuring"
      readonly model: LocalModel
    }
  | {
      readonly _tag: "Activating"
      readonly providerModelId: ProviderModelId
      readonly displayName: string
      readonly phase: "Loading" | "Stopping" | "Ready" | "Failed"
      readonly failure: ModelInstanceFailure | null
      readonly onRetry: () => void
      readonly onChooseAnother: () => void
    }

export function OnboardingModelChooser({
  hardware,
  options,
  width,
  error,
  operation,
  onSelect,
  onSkip,
}: {
  readonly hardware: LocalInferenceHardwareResult
  readonly options: readonly LocalModelOption[]
  readonly width: number
  readonly error: string | null
  readonly operation: OnboardingModelChooserOperation | null
  readonly onSelect: (configurationId: ModelServingConfigurationId) => void
  readonly onSkip: () => void
}): ReactNode {
  const theme = useTheme()
  const selections = useMemo(() =>
    options.filter((selection) =>
      selection.kind !== "recommendation"
        || Option.isSome(selection.recommendation)),
  [options])
  const [selectedId, setSelectedId] = useState<Option.Option<string>>(Option.none())
  const localScrollRef = useRef<ScrollBoxRenderable | null>(null)
  const downloadScrollRef = useRef<ScrollBoxRenderable | null>(null)
  const activeSelectionId = operation === null
    ? Option.none<string>()
    : Option.fromNullable(selections.find((selection) =>
      operation._tag === "Downloading" || operation._tag === "Configuring"
      ? Option.exists(localModelConfigurationId(operation.model), (configurationId) =>
          Option.contains(selectionConfigurationId(selection), configurationId))
      : Option.contains(selectionProviderModelId(selection), operation.providerModelId))?.id)
  const selectedIndex = selectedInferenceIndex(
    selections,
    Option.isSome(activeSelectionId) ? activeSelectionId : selectedId,
  )
  const selected = selections[selectedIndex]
  const selectedMemory = selected?.model.servingState._tag === "Assessed"
    && selected.model.servingState.assessment._tag === "Fits"
    ? Option.some(selected.model.servingState.assessment.memory)
    : Option.none<LocalModelMemory>()
  const locked = operation !== null
  const local = selections.filter(({ kind }) => kind === "running" || kind === "stored")
  const downloads = selections.filter(({ kind }) => kind === "recommendation")
  const cardWidth = setupCardWidth(width)
  const wide = cardWidth >= 82
  const leftWidth = wide ? WIDE_LIST_WIDTH : Math.max(1, cardWidth - 6)
  const detailWidth = wide ? Math.max(1, cardWidth - leftWidth - 9) : leftWidth
  const localRows = local.length > 0 ? SECTION_VIEWPORT_ROWS + 1 : 0
  const downloadRows = downloads.length > 0 ? SECTION_VIEWPORT_ROWS + 1 : 0
  const sectionGap = local.length > 0 && downloads.length > 0 ? 1 : 0
  const selectedMemoryRows = Option.match(selectedMemory, {
    onNone: () => 0,
    onSome: memoryGuidanceRows,
  })
  const systemUsedBytes = Result.isSuccess(hardware)
    ? Math.max(0, hardware.value.totalSystemMemoryBytes - hardware.value.availableSystemMemoryBytes)
    : null
  const contentHeight = Math.max(
    DETAIL_BASE_ROWS + selectedMemoryRows,
    localRows + sectionGap + downloadRows,
  )
  const detailContentHeight = Math.max(1, contentHeight - (wide ? 0 : 1))
  const choose = useCallback((selection: LocalInferenceSelection) => {
    const configurationId = selectionConfigurationId(selection)
    if (Option.isSome(configurationId)) onSelect(configurationId.value)
  }, [onSelect])

  const moveSelectionTo = useCallback((index: number) => {
    const selection = selections[index]
    if (!selection) return
    setSelectedId(Option.some(selection.id))
    scrollOnboardingModelIntoView(
      selection.kind === "recommendation" ? downloadScrollRef.current : localScrollRef.current,
      selection.id,
    )
  }, [selections])

  useKeyboard(useCallback((key: KeyEvent) => {
    if (locked) {
      key.preventDefault()
      return
    }
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
      onSkip()
    }
  }, [choose, locked, moveSelectionTo, onSkip, selected, selectedIndex, selections.length]))

  const list = (
    <box style={{ width: wide ? leftWidth : "100%", flexDirection: "column", paddingRight: wide ? 1 : 0 }}>
      {local.length > 0 && <text style={{ fg: theme.muted }} attributes={TextAttributes.BOLD}>ON THIS COMPUTER</text>}
      {local.length > 0 && (
        <ModelSectionViewport scrollRef={localScrollRef}>
          {local.map((selection) => (
            <ModelRow
              key={selection.id}
              selection={selection}
              selected={selection.id === selected?.id}
              disabled={locked}
              width={leftWidth}
              rowId={onboardingModelRowId(selection.id)}
              onHover={() => setSelectedId(Option.some(selection.id))}
              onChoose={() => choose(selection)}
            />
          ))}
        </ModelSectionViewport>
      )}
      {downloads.length > 0 && (
        <text style={{ fg: theme.muted, marginTop: local.length > 0 ? 1 : 0 }} attributes={TextAttributes.BOLD}>
          AVAILABLE TO DOWNLOAD
        </text>
      )}
      {downloads.length > 0 && (
        <ModelSectionViewport scrollRef={downloadScrollRef}>
          {downloads.map((selection) => (
            <ModelRow
              key={selection.id}
              selection={selection}
              selected={selection.id === selected?.id}
              disabled={locked}
              width={leftWidth}
              rowId={onboardingModelRowId(selection.id)}
              onHover={() => setSelectedId(Option.some(selection.id))}
              onChoose={() => choose(selection)}
            />
          ))}
        </ModelSectionViewport>
      )}
    </box>
  )

  const recommendationIntent = selected && Option.isSome(selected.recommendation)
    ? intentLabel(selected.recommendation.value.intent)
    : null
  const titleNameWidth = Math.max(
    1,
    detailWidth - (recommendationIntent ? recommendationIntent.length + 3 : 0),
  )
  const emptySelectionMessage = "No compatible models found."
  const regularDetails = selected ? (
    <>
      <DetailRow width={detailWidth}>
        <text
          style={{ fg: theme.foreground, width: detailWidth }}
          attributes={TextAttributes.BOLD}
          wrapMode="none"
        >
          {truncateToDisplayWidth(selected.model.presentation.displayName, titleNameWidth)}
          {recommendationIntent && <span fg={theme.primary}>{`   ${recommendationIntent}`}</span>}
        </text>
      </DetailRow>
      <DetailRow width={detailWidth}>
        <text style={{ fg: theme.muted, width: detailWidth }} wrapMode="none">
          {selectionMetadata(selected)}
        </text>
      </DetailRow>
      {Option.isSome(selected.recommendation) && (
        <DetailRow width={detailWidth}>
          <text style={{ fg: theme.muted, width: detailWidth }} wrapMode="none">
            {performanceRangeSpeedLabel(selected.model)}
          </text>
        </DetailRow>
      )}
      <box style={{ height: 1, flexShrink: 0 }} />
      <box style={{
        width: detailWidth,
        maxHeight: DESCRIPTION_ROWS,
        flexShrink: 0,
        flexDirection: "column",
        overflow: "hidden",
      }}>
        <text style={{ fg: theme.muted, width: detailWidth }} wrapMode="word">
          {Option.isSome(selected.recommendation)
            ? selected.recommendation.value.explanation
            : selected.kind === "running"
              ? "Loaded in memory and ready to use."
              : "Downloaded on this computer and ready to load."}
        </text>
      </box>
      {Option.isSome(selectedMemory) && (
        <>
          <box style={{ height: 1, flexShrink: 0 }} />
          <ModelMemoryGuidanceDetails
            memory={selectedMemory.value}
            systemUsedBytes={systemUsedBytes}
            width={detailWidth}
          />
        </>
      )}
    </>
  ) : (
    <text style={{ fg: theme.muted }}>{emptySelectionMessage}</text>
  )
  const detailsContent = operation?._tag === "Downloading" ? (
    <OnboardingModelDownloadDetails
      model={operation.model}
      width={detailWidth}
      height={detailContentHeight}
      operation={{
        starting: operation.starting,
        cancelling: operation.cancelling,
        cancelError: operation.cancelError,
        onCancel: operation.onCancel,
      }}
    />
  ) : operation?._tag === "Activating" ? (
    <OnboardingModelLoadingDetails
      displayName={operation.displayName}
      width={detailWidth}
      height={detailContentHeight}
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
      height: contentHeight,
      minHeight: contentHeight,
      maxHeight: contentHeight,
      overflow: "hidden",
      paddingLeft: wide ? 2 : 0,
      paddingTop: wide ? 0 : 1,
      borderStyle: "single",
      border: wide ? ["left"] : ["top"],
      borderColor: theme.border,
      customBorderChars: BOX_CHARS,
    }}>
      {detailsContent}
    </box>
  )
  const interactionHint = operation?._tag === "Downloading"
      ? "Download in progress · Esc cancel"
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
        ? "Close memory-intensive apps, then Enter to retry · Esc choose another"
        : "Close memory-intensive apps before loading · Enter select · Esc skip for now"
      : "↑/↓ choose · Enter select · Esc skip for now"

  return (
    <OnboardingSetupCard
      cardWidth={cardWidth}
      title="Choose a local model"
      hardware={hardware}
    >
      <box style={{
        flexDirection: wide ? "row" : "column",
        width: "100%",
        height: wide ? contentHeight : contentHeight * 2,
        minHeight: wide ? contentHeight : contentHeight * 2,
        maxHeight: wide ? contentHeight : contentHeight * 2,
        overflow: "hidden",
      }}>
        {list}
        {details}
      </box>
      {error && <text style={{ fg: theme.error, marginTop: 1 }}>{error}</text>}
      <box style={{ height: 1 }} />
      <text style={{ fg: slate[200] }}>You can switch models or download more anytime from /settings.</text>
      <text style={{ fg: theme.muted }}>{interactionHint}</text>
    </OnboardingSetupCard>
  )
}

export function OnboardingModelPreparation({
  hardware,
  progress,
  error,
  width,
  onSkip,
}: {
  readonly hardware: LocalInferenceHardwareResult
  readonly progress: readonly LocalModelRecommendationProgressStep[]
  readonly error: string | null
  readonly width: number
  readonly onSkip: () => void
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
    if (key.name === "escape") {
      key.preventDefault()
      onSkip()
    }
  }, [onSkip]))
  return (
    <OnboardingSetupCard
      cardWidth={cardWidth}
      title="Preparing local models"
      hardware={hardware}
      spinnerFrame={spinner}
    >
      {lines.map((line) => (
        <text key={line.id} style={{ fg: line.state === "pending" ? theme.muted : theme.foreground }}>
          <span fg={line.state === "completed" ? theme.success : line.state === "failed" ? theme.error : line.state === "running" ? theme.primary : theme.muted}>
            {line.state === "completed" ? "✓ " : line.state === "failed" ? "! " : line.state === "running" ? `${spinner} ` : "○ "}
          </span>
          {line.label}<span fg={line.state === "failed" ? theme.error : theme.muted}>{line.metadata}</span>
        </text>
      ))}
      {error && <text style={{ fg: theme.error }}>{error}</text>}
      <box style={{ height: 1 }} />
      <text style={{ fg: theme.muted }}>Esc skip for now</text>
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
        <text style={{ fg: theme.foreground, width }} attributes={TextAttributes.BOLD} wrapMode="none">
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
              <text style={{ fg: theme.warning, width }} attributes={TextAttributes.BOLD}>
                ! Not enough memory available
              </text>
              <box style={{ height: 1 }} />
              <text style={{ fg: theme.foreground, width }} wrapMode="word">
                {`Free at least ${minimumBytesLabel(failed.minimumAdditionalAvailableBytes)} and try again.`}
              </text>
              <text style={{ fg: theme.muted, width }} wrapMode="word">
                Close memory-intensive applications or choose a smaller model.
              </text>
              <box style={{ height: 1 }} />
              <text style={{ fg: theme.muted, width }}>
                {`Needed at attempt    ${formatBytes(failed.loadBoundaryBytes)}`}
              </text>
              <text style={{ fg: theme.muted, width }}>
                {`Available at attempt ${formatBytes(failed.allocationHeadroomBytes)}`}
              </text>
              <box style={{ height: 1 }} />
            </box>
          ) : (
            <box style={{ width, height: 5, flexShrink: 0, flexDirection: "column", overflow: "hidden" }}>
              <text style={{ fg: theme.error, width }} wrapMode="word">{failed.message}</text>
            </box>
          )}
          <box style={{ flexDirection: "row", gap: 2 }}>
            <Button onClick={onRetry} onMouseOver={() => setHovered("retry")} onMouseOut={() => setHovered(null)}>
              <text style={{ fg: hovered === "retry" ? theme.primary : theme.foreground }}>Retry loading</text>
            </Button>
            <Button onClick={onChooseAnother} onMouseOver={() => setHovered("choose")} onMouseOut={() => setHovered(null)}>
              <text style={{ fg: hovered === "choose" ? theme.primary : theme.foreground }}>Choose another model</text>
            </Button>
          </box>
        </>
      ) : (
        <box style={{ width, flexDirection: "row" }}>
          <text style={{ fg: theme.primary, width: 2, flexShrink: 0 }} wrapMode="none">
            {spinner}
          </text>
          <text style={{ fg: theme.muted, width: Math.max(1, width - 2) }} wrapMode="none">
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
