import { useCallback, useMemo, useRef, useState, type ReactNode, type Ref } from "react"
import { TextAttributes, type KeyEvent, type ScrollBoxRenderable } from "@opentui/core"
import { useKeyboard } from "@opentui/react"
import { Result } from "@effect-atom/atom-react"
import { Option } from "effect"
import {
  truncateToDisplayWidth,
  type LocalInferenceHardwareResult,
  type OnboardingConfigurationChoice,
  type OnboardingLoadModelChoice,
} from "@magnitudedev/client-common"
import type {
  LocalModelCatalogCandidate,
  LocalModelRecommendationProgressStep,
  LocalModelsState,
  ModelSlotsState,
  ProviderModelId,
  ProviderModelCatalogState,
} from "@magnitudedev/sdk"
import { ReasoningEffortSchema } from "@magnitudedev/sdk"
import { Button } from "../../components/button"
import { spinnerFrameAt, useSpinnerFrame } from "../../hooks/use-spinner-frame"
import { useTheme } from "../../hooks/use-theme"
import { BOX_CHARS } from "../../utils/ui-constants"
import {
  buildLocalInferenceSelections,
  describeLocalHardwareSummary,
  localInferenceProgressLines,
  performanceRangeSpeedLabel,
  selectedInferenceIndex,
  selectionCapacityWarning,
  selectionMetadata,
  type LocalInferenceSelection,
} from "../local-inference/view-model"
import { slate } from "../../utils/theme"
import { OnboardingModelDownloadDetails } from "./download-details"

const SECTION_VIEWPORT_ROWS = 4
const DESCRIPTION_ROWS = 5
const DETAIL_FIXED_ROWS = 10
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
    return selection.recommendation._tag === "Recommended"
      ? intentLabel(selection.recommendation.value.intent)
      : "Download"
  }
  return "Load"
}

const onboardingSelection = (
  selection: LocalInferenceSelection,
): ProviderModelId | null => Option.getOrNull(selection.providerModelId)

const matchesOnboardingSelection = (
  selection: LocalInferenceSelection,
  submitted: ProviderModelId,
): boolean => Option.contains(selection.providerModelId, submitted)

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
    || selection.recommendation._tag === "Recommended"
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
        {selected ? "› " : "  "}{truncateToDisplayWidth(selection.model.displayName, nameWidth).padEnd(nameWidth)}
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
      readonly candidate: LocalModelCatalogCandidate
      readonly cancelling: boolean
      readonly cancelError: string | null
      readonly onCancel: () => void
      readonly onRetry: () => void
    }
  | {
      readonly _tag: "DownloadFailed"
      readonly candidate: LocalModelCatalogCandidate
      readonly onChooseAnother: () => void
      readonly onRetry: () => void
    }
  | {
      readonly _tag: "Configuring"
      readonly candidate: LocalModelCatalogCandidate
    }
  | {
      readonly _tag: "Activating"
      readonly providerModelId: ProviderModelId
      readonly displayName: string
      readonly phase: "Loading" | "Stopping" | "Ready" | "Failed"
      readonly failure: string | null
      readonly onRetry: () => void
      readonly onChooseAnother: () => void
    }

export function OnboardingModelChooser({
  hardware,
  models,
  catalog,
  slots,
  width,
  error,
  operation,
  onLoad,
  onSelectConfiguration,
  onContinue,
  onSkip,
}: {
  readonly hardware: LocalInferenceHardwareResult
  readonly models: LocalModelsState
  readonly catalog: ProviderModelCatalogState
  readonly slots: ModelSlotsState
  readonly width: number
  readonly error: string | null
  readonly operation: OnboardingModelChooserOperation | null
  readonly onLoad: (choice: OnboardingLoadModelChoice) => void
  readonly onSelectConfiguration: (choice: OnboardingConfigurationChoice) => void
  readonly onContinue: () => void
  readonly onSkip: () => void
}): ReactNode {
  const theme = useTheme()
  const selections = useMemo(() =>
    buildLocalInferenceSelections(models, catalog, slots).filter((selection) =>
      selection.kind !== "recommendation"
        || selection.recommendation._tag === "Recommended"),
  [catalog, models, slots])
  const [selectedId, setSelectedId] = useState<Option.Option<string>>(Option.none())
  const localScrollRef = useRef<ScrollBoxRenderable | null>(null)
  const downloadScrollRef = useRef<ScrollBoxRenderable | null>(null)
  const selectionConfigurationId = (selection: LocalInferenceSelection) =>
    selection.kind === "recommendation"
      ? selection.recommendation._tag === "Recommended"
        ? selection.recommendation.value.candidate.configurationId
        : undefined
      : selection.configurationId
  const activeSelectionId = operation === null
    ? Option.none<string>()
    : Option.fromNullable(selections.find((selection) =>
      operation._tag === "Downloading" || operation._tag === "DownloadFailed"
        || operation._tag === "Configuring"
      ? selectionConfigurationId(selection) === operation.candidate.configurationId
      : Option.contains(selection.providerModelId, operation.providerModelId))?.id)
  const selectedIndex = selectedInferenceIndex(
    selections,
    Option.isSome(activeSelectionId) ? activeSelectionId : selectedId,
  )
  const selected = selections[selectedIndex]
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
  const contentHeight = Math.max(DETAIL_FIXED_ROWS, localRows + sectionGap + downloadRows)
  const detailContentHeight = Math.max(1, contentHeight - (wide ? 0 : 1))
  const choose = useCallback((selection: LocalInferenceSelection) => {
    if (selection.kind === "running") {
      onContinue()
      return
    }
    if (selection.kind === "stored" && Option.isSome(selection.providerModelId)) {
      onLoad({
        providerModelId: selection.providerModelId.value,
        displayName: selection.model.displayName,
        reasoningEffort: Option.getOrElse(
          selection.reasoningEffort,
          () => ReasoningEffortSchema.make("none"),
        ),
      })
      return
    }
    if (selection.kind === "stored") {
      onSelectConfiguration({
        targetId: selection.model.targetId,
        configurationId: selection.configurationId,
        displayName: selection.model.displayName,
        reasoningEffort: Option.getOrElse(
          selection.reasoningEffort,
          () => ReasoningEffortSchema.make("none"),
        ),
      })
      return
    }
    if (
      selection.kind === "recommendation"
      && selection.recommendation._tag === "Recommended"
    ) {
      const candidate = selection.recommendation.value.candidate
      onSelectConfiguration({
        targetId: candidate.targetId,
        configurationId: candidate.configurationId,
        displayName: candidate.displayName,
        reasoningEffort: Option.getOrElse(
          selection.reasoningEffort,
          () => ReasoningEffortSchema.make("none"),
        ),
      })
    }
  }, [onContinue, onLoad, onSelectConfiguration])

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

  const recommendationIntent = selected?.recommendation._tag === "Recommended"
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
          {truncateToDisplayWidth(selected.model.displayName, titleNameWidth)}
          {recommendationIntent && <span fg={theme.primary}>{`   ${recommendationIntent}`}</span>}
        </text>
      </DetailRow>
      <DetailRow width={detailWidth}>
        <text style={{ fg: theme.muted, width: detailWidth }} wrapMode="none">
          {selectionMetadata(selected)}
        </text>
      </DetailRow>
      <DetailRow width={detailWidth}>
        {selected.recommendation._tag === "Recommended" && (
          <text style={{ fg: theme.muted, width: detailWidth }} wrapMode="none">
            {performanceRangeSpeedLabel(selected.recommendation.value.candidate)}
          </text>
        )}
      </DetailRow>
      <box style={{ height: 1 }} />
      <box style={{
        width: detailWidth,
        height: DESCRIPTION_ROWS,
        minHeight: DESCRIPTION_ROWS,
        maxHeight: DESCRIPTION_ROWS,
        flexShrink: 0,
        flexDirection: "column",
        overflow: "hidden",
      }}>
        <text style={{ fg: theme.muted, width: detailWidth }} wrapMode="word">
          {selected.recommendation._tag === "Recommended"
            ? selected.recommendation.value.explanation
            : selected.kind === "running"
              ? "Loaded in memory and ready to use."
              : "Downloaded on this computer and ready to load."}
        </text>
      </box>
      <DetailRow width={detailWidth}>
        {selectionCapacityWarning(selected) && (
          <text style={{ fg: theme.warning, width: detailWidth }} wrapMode="none">
            {selectionCapacityWarning(selected)}
          </text>
        )}
      </DetailRow>
    </>
  ) : (
    <text style={{ fg: theme.muted }}>{emptySelectionMessage}</text>
  )
  const detailsContent = operation?._tag === "Downloading" ? (
    <OnboardingModelDownloadDetails
      candidate={operation.candidate}
      width={detailWidth}
      height={detailContentHeight}
      operation={{
        _tag: "Active",
        cancelling: operation.cancelling,
        cancelError: operation.cancelError,
        onCancel: operation.onCancel,
        onRetry: operation.onRetry,
      }}
    />
  ) : operation?._tag === "DownloadFailed" ? (
    <OnboardingModelDownloadDetails
      candidate={operation.candidate}
      width={detailWidth}
      height={detailContentHeight}
      operation={{
        _tag: "Failed",
        onChooseAnother: operation.onChooseAnother,
        onRetry: operation.onRetry,
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
      : operation?._tag === "DownloadFailed"
        ? "Download failed · Retry or choose another model"
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
  readonly failed: string | null
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
          <box style={{ width, height: 5, flexShrink: 0, flexDirection: "column", overflow: "hidden" }}>
            <text style={{ fg: theme.error, width }} wrapMode="word">{failed}</text>
          </box>
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
