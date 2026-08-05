import {
  memo,
  useCallback,
  useMemo,
  useState,
  type ReactNode,
} from "react"
import { TextAttributes, type KeyEvent } from "@opentui/core"
import { useKeyboard } from "@opentui/react"
import { Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Cause, Option } from "effect"
import {
  deriveHardwareMemoryView,
  deriveCurrentLocalModel,
  modelMemoryStatusDetail,
  modelMemoryStatusLabel,
  modelSlotInstanceId,
  modelSlotResidentAllocation,
  requiredMemoryBytes,
  providerModelMemoryConditions,
  selectedSlotModel,
  usePlatform,
  useLocalInferenceHardware,
  useLocalModelActions,
  useLocalModels,
  useModelSlotActions,
  usePreviewModelLoad,
  useModelConfig,
  useSettingsState,
} from "@magnitudedev/client-common"
import {
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelCatalogLifecycle,
  ReasoningEffortSchema,
  type LocalModel,
  type LocalModelCatalogCandidate,
  type LocalModelRecommendation,
  type ProviderModelId,
  type ProviderModelCatalogEntry,
} from "@magnitudedev/sdk"
import { Button } from "../../components/button"
import { HardwareMemoryDomain } from "../../components/hardware-memory-domain"
import { useSpinnerFrame } from "../../hooks/use-spinner-frame"
import { useBoundedCursor } from "../../hooks/use-bounded-cursor"
import { useTheme } from "../../hooks/use-theme"
import {
  authSourceAtom,
  modelMenuStateAtom,
  type ModelMenuRoot,
} from "../../state/cli-atoms"
import { SingleLineInput } from "../composer/single-line-input"
import {
  describeLocalHardware,
  formatBytes,
  localInferenceProgressLines,
  performanceRangeSpeedLabel,
} from "../local-inference/view-model"
import { deriveSettingsAuthInfo } from "../overlays/auth-display"

const ROOTS = ["models", "catalog", "hardware", "cloud"] as const
const ROOT_LABELS: Record<ModelMenuRoot, string> = {
  models: "MODELS",
  catalog: "CATALOG",
  hardware: "HARDWARE",
  cloud: "CLOUD",
}
const MENU_HEIGHT = 32
const LOCAL_PROVIDER_ID = ProviderIdSchema.make("local")
const MAGNITUDE_CLOUD_URL = "https://app.magnitude.dev"
type CloudActionId = "add" | "update" | "disconnect" | "link"
const EMPTY_MODEL_ACTIONS = [
  { label: "Find a local model", root: "catalog" },
  { label: "Connect cloud models", root: "cloud" },
] as const satisfies readonly { readonly label: string; readonly root: ModelMenuRoot }[]

interface ModelsMenuProps {
  readonly openRoot: (root: ModelMenuRoot) => void
  readonly openCatalogDetail: (providerModelId: string) => void
  readonly setRootSwitchingEnabled: (enabled: boolean) => void
}

interface CatalogMenuProps {
  readonly initialCatalogDetailId: string | null
  readonly setRootSwitchingEnabled: (enabled: boolean) => void
}

interface CloudMenuProps {
  readonly setRootSwitchingEnabled: (enabled: boolean) => void
}

const nextRoot = (root: ModelMenuRoot, direction: -1 | 1): ModelMenuRoot => {
  const index = ROOTS.indexOf(root)
  return ROOTS[(index + direction + ROOTS.length) % ROOTS.length]!
}

export const resolveRootNavigationDirection = (
  key: Pick<KeyEvent, "name" | "ctrl" | "meta" | "option" | "shift">,
): -1 | 1 | null => {
  if (key.ctrl || key.meta || key.option) return null
  if (key.name === "left") return -1
  if (key.name === "right") return 1
  if (key.name === "tab") return key.shift ? -1 : 1
  return null
}

const formatContextWindow = (tokens: number): string =>
  tokens >= 1_000_000
    ? `${(tokens / 1_000_000).toFixed(1)}M`
    : tokens >= 1_000
      ? `${Math.round(tokens / 1_000)}K`
      : String(tokens)

const providerModelKey = (model: Pick<ProviderModelCatalogEntry, "providerId" | "providerModelId">): string =>
  `${model.providerId}:${model.providerModelId}`

const catalogModels = (
  config: ReturnType<typeof useModelConfig>,
): readonly ProviderModelCatalogEntry[] => Option.getOrElse(
  Option.flatMap(Result.value(config.catalog), ({ state }) =>
    ProviderModelCatalogLifecycle.match(state, {
      Loading: () => Option.none(),
      Ready: ({ models }) => Option.some(models),
      Refreshing: ({ models }) => Option.some(models),
      Degraded: ({ models }) => Option.some(models),
      Unavailable: () => Option.none(),
    })),
  () => [],
)

export function ModelMenusContainer({
  downloadSummary,
}: {
  readonly downloadSummary: string | null
}): ReactNode {
  const menu = useAtomValue(modelMenuStateAtom)
  const setMenu = useAtomSet(modelMenuStateAtom)
  const theme = useTheme()
  const [atRootLevel, setAtRootLevel] = useState(true)
  const [hoveredRoot, setHoveredRoot] = useState<ModelMenuRoot | null>(null)
  const [catalogDetailId, setCatalogDetailId] = useState<string | null>(null)
  const openRoot = useCallback((root: ModelMenuRoot) => {
    setCatalogDetailId(null)
    setAtRootLevel(true)
    setMenu({ open: true, root })
  }, [setMenu])
  const openCatalogDetail = useCallback((providerModelId: string) => {
    setCatalogDetailId(providerModelId)
    setAtRootLevel(false)
    setMenu({ open: true, root: "catalog" })
  }, [setMenu])
  const close = useCallback(() => {
    setMenu((current) => ({ ...current, open: false }))
  }, [setMenu])

  useKeyboard(useCallback((key: KeyEvent) => {
    if (!menu.open || key.defaultPrevented) return
    const rootNavigationDirection = resolveRootNavigationDirection(key)
    if (rootNavigationDirection !== null) {
      key.preventDefault()
      openRoot(nextRoot(menu.root, rootNavigationDirection))
      return
    }
    if (atRootLevel && key.name === "escape") {
      key.preventDefault()
      close()
    }
  }, [atRootLevel, close, menu.open, menu.root, openRoot]))

  if (!menu.open) return null

  return (
    <box
      style={{
        height: MENU_HEIGHT,
        maxHeight: "100%",
        minHeight: 0,
        width: "100%",
        flexShrink: 0,
        flexDirection: "column",
        backgroundColor: "transparent",
      }}
    >
      <box style={{ flexGrow: 1, minHeight: 0, flexDirection: "column", backgroundColor: theme.menuBg }}>
        {menu.root === "models" && <ModelsMenu openRoot={openRoot} openCatalogDetail={openCatalogDetail} setRootSwitchingEnabled={setAtRootLevel} />}
        {menu.root === "catalog" && <CatalogMenu initialCatalogDetailId={catalogDetailId} setRootSwitchingEnabled={setAtRootLevel} />}
        {menu.root === "hardware" && <HardwareMenu />}
        {menu.root === "cloud" && <CloudMenu setRootSwitchingEnabled={setAtRootLevel} />}
      </box>
      <box
        style={{
          height: 1,
          flexShrink: 0,
          borderStyle: "single",
          border: ["bottom"],
          borderColor: theme.menuBg,
          customBorderChars: {
            topLeft: "",
            bottomLeft: "",
            topRight: "",
            bottomRight: "",
            horizontal: "▀",
            vertical: " ",
            topT: "",
            bottomT: "",
            leftT: "",
            rightT: "",
            cross: "",
          },
        }}
      />
      <box
        style={{
          flexShrink: 0,
          flexDirection: "row",
          alignItems: "center",
          backgroundColor: "transparent",
          paddingLeft: 1,
          paddingRight: 1,
          height: 1,
        }}
      >
        {ROOTS.map((root) => {
          const active = root === menu.root
          return (
            <Button
              key={root}
              onClick={() => openRoot(root)}
              onMouseOver={() => setHoveredRoot(root)}
              onMouseOut={() => setHoveredRoot(null)}
              style={{ marginRight: 2 }}
            >
              <text
                style={{
                  fg: active ? theme.menuBg : theme.foreground,
                  ...(active ? { bg: theme.foreground } : {}),
                }}
                attributes={active ? TextAttributes.BOLD : TextAttributes.NONE}
              >
                {" "}
                <span attributes={hoveredRoot === root && !active ? TextAttributes.UNDERLINE : TextAttributes.NONE}>
                  {ROOT_LABELS[root]}
                </span>
                {" "}
              </text>
            </Button>
          )
        })}
        {downloadSummary && (
          <Button onClick={() => openRoot("catalog")}>
            <text style={{ fg: theme.primary }}>{downloadSummary}</text>
          </Button>
        )}
        <box style={{ flexGrow: 1 }} />
        <text style={{ fg: theme.muted }}>
          {atRootLevel ? "←/→ switch menus" : "←/→ switch menus · Esc back"}
        </text>
      </box>
    </box>
  )
}

const MenuHeader = memo(function MenuHeader({
  title,
  subtitle,
  selection,
  onSectionClick,
  summary,
  hints,
}: {
  readonly title: string
  readonly subtitle?: string
  readonly selection?: string
  readonly onSectionClick?: () => void
  readonly summary?: string
  readonly hints?: string
}) {
  const theme = useTheme()
  const [sectionHovered, setSectionHovered] = useState(false)
  const sectionTitle = (
    <text
      style={{ fg: theme.foreground }}
      attributes={TextAttributes.BOLD | (sectionHovered ? TextAttributes.UNDERLINE : TextAttributes.NONE)}
    >
      {title.toUpperCase()}
    </text>
  )
  return (
    <box style={{ flexShrink: 0, flexDirection: "column", paddingLeft: 2, paddingRight: 2, paddingTop: 1 }}>
      <box style={{ flexDirection: "row" }}>
        {selection && onSectionClick ? (
          <Button
            onClick={onSectionClick}
            onMouseOver={() => setSectionHovered(true)}
            onMouseOut={() => setSectionHovered(false)}
          >
            {sectionTitle}
          </Button>
        ) : sectionTitle}
        {subtitle && <text style={{ fg: theme.muted }}> · {subtitle}</text>}
        {selection && <text style={{ fg: theme.foreground }}> → {selection}</text>}
        <box style={{ flexGrow: 1 }} />
        {summary && <text style={{ fg: theme.muted }}>{summary}</text>}
      </box>
      {hints && <text style={{ fg: theme.muted }}>{hints}</text>}
    </box>
  )
})

type MenuActionTone = "primary" | "normal" | "link" | "warning" | "error"

const MenuAction = memo(function MenuAction({
  label,
  focused,
  tone = "normal",
  onClick,
  onMouseOver,
}: {
  readonly label: string
  readonly focused: boolean
  readonly tone?: MenuActionTone
  readonly onClick: () => void
  readonly onMouseOver: () => void
}) {
  const theme = useTheme()
  const color = focused
    ? theme.primary
    : tone === "primary"
      ? theme.primary
      : tone === "link"
        ? theme.link
        : tone === "warning"
          ? theme.warning
          : tone === "error"
            ? theme.error
            : theme.foreground
  return (
    <Button onClick={onClick} onMouseOver={onMouseOver}>
      <text style={{ fg: color }}>{focused ? "› " : "  "}{label}</text>
    </Button>
  )
})

const ModelsMenu = memo(function ModelsMenu({
  openRoot,
  openCatalogDetail,
  setRootSwitchingEnabled,
}: ModelsMenuProps) {
  const theme = useTheme()
  const config = useModelConfig()
  const localModels = useLocalModels()
  const slotActions = useModelSlotActions()
  const hardware = Option.getOrUndefined(Result.value(useLocalInferenceHardware()))
  const models = catalogModels(config)
  const catalogSnapshot = Result.value(config.catalog)
  const slotsSnapshot = Result.value(config.slots)
  const selected = Option.flatMap(
    Option.all({ catalog: catalogSnapshot, slots: slotsSnapshot }),
    ({ catalog, slots }) => selectedSlotModel(catalog.state, slots.state, PRIMARY_SLOT_ID),
  )
  const selectedKey = Option.match(selected, {
    onNone: () => null,
    onSome: ({ model }) => providerModelKey(model),
  })
  const currentRecentModelIds = Option.match(slotsSnapshot, {
    onNone: () => [] as readonly string[],
    onSome: ({ state }) => state.recentModelIds.primary,
  })
  const currentFavoriteKeys = new Set(config.favoriteModels.map(providerModelKey))
  const [ordering] = useState(() => ({
    selectedKey,
    recentModelIds: currentRecentModelIds,
    favoriteKeys: currentFavoriteKeys,
  }))
  const eligible = models
    .filter((model) =>
      model.supportedSlots.includes(PRIMARY_SLOT_ID)
      && (model.availability._tag === "Available"
        || providerModelKey(model) === selectedKey
        || (model.providerId === LOCAL_PROVIDER_ID
          && model.availability.reason === "insufficient_resources")))
    .sort((left, right) => {
      const leftFavorite = ordering.favoriteKeys.has(providerModelKey(left))
      const rightFavorite = ordering.favoriteKeys.has(providerModelKey(right))
      if (leftFavorite !== rightFavorite) return leftFavorite ? -1 : 1
      const leftSelected = providerModelKey(left) === ordering.selectedKey
      const rightSelected = providerModelKey(right) === ordering.selectedKey
      if (leftSelected !== rightSelected) return leftSelected ? -1 : 1
      const leftRecency = ordering.recentModelIds.indexOf(left.providerModelId)
      const rightRecency = ordering.recentModelIds.indexOf(right.providerModelId)
      if (leftRecency !== rightRecency) {
        if (leftRecency < 0) return 1
        if (rightRecency < 0) return -1
        return leftRecency - rightRecency
      }
      const leftLocal = left.providerId === LOCAL_PROVIDER_ID
      const rightLocal = right.providerId === LOCAL_PROVIDER_ID
      if (leftLocal !== rightLocal) return leftLocal ? -1 : 1
      return left.displayName.localeCompare(right.displayName)
    })
  const [cursorId, setCursorId] = useState<string | null>(null)
  const [detailId, setDetailId] = useState<string | null>(null)
  const cursorIndex = Math.max(0, eligible.findIndex((model) => providerModelKey(model) === cursorId))
  const cursor = eligible[cursorIndex]
  const detail = eligible.find((model) => providerModelKey(model) === detailId) ?? null
  const localSnapshot = Result.value(localModels)
  const localCatalogCandidates = Option.match(localSnapshot, {
    onNone: () => [] as readonly LocalModelCatalogCandidate[],
    onSome: (models) =>
      models.recommendations._tag === "Ready" ? models.recommendations.catalog : [],
  })
  const requirementFor = (model: ProviderModelCatalogEntry): string => {
    if (model.providerId !== LOCAL_PROVIDER_ID) return "Cloud"
    return Option.match(model.memory, {
      onNone: () => "—",
      onSome: (memory) => formatBytes(requiredMemoryBytes(memory)),
    })
  }
  const assessingRequirementFor = (model: LocalModel): string => {
    const candidate = localCatalogCandidates.find(({ targetId }) => targetId === model.targetId)
    return candidate ? formatBytes(requiredMemoryBytes(candidate.memory)) : "—"
  }
  const assessing = Option.match(localSnapshot, {
    onNone: () => [] as readonly LocalModel[],
    onSome: (models) => models.models.filter((model) =>
      model.download._tag === "Downloaded" && model.assessment._tag === "Assessing"),
  })
  const primarySlot = Option.match(slotsSnapshot, {
    onNone: () => null,
    onSome: ({ state }) => state.slots.primary,
  })
  const residentAllocation = primarySlot === null
    ? Option.none()
    : modelSlotResidentAllocation(primarySlot)
  const detailIsLocal = detail?.providerId === LOCAL_PROVIDER_ID
  const detailIsSelected = detail !== null && providerModelKey(detail) === selectedKey
  const detailLocalModel = detailIsLocal && detail
    ? Option.match(localSnapshot, {
        onNone: () => undefined,
        onSome: (models) => models.models.find(({ offerings }) =>
          offerings.some(({ providerModelId }) => providerModelId === detail.providerModelId)),
      })
    : undefined
  const detailCatalogCandidate = detailLocalModel && detail
    ? localCatalogCandidates.find(({ configurationId }) =>
        detailLocalModel.offerings.some((offering) =>
          offering.providerModelId === detail.providerModelId
          && offering.configurationId === configurationId))
    : undefined
  const detailActions = useMemo(() => {
    if (!detail) return [] as readonly ("select" | "load" | "stop" | "catalog")[]
    const actions: ("select" | "load" | "stop" | "catalog")[] = []
    const memoryConditions = providerModelMemoryConditions(detail, hardware, residentAllocation)
    if (!detailIsSelected
      && detail.availability._tag === "Available"
      && detail.supportedSlots.includes(PRIMARY_SLOT_ID)) actions.push("select")
    if (detailIsLocal
      && detailIsSelected
      && primarySlot?._tag === "ConfiguredLocal"
      && primarySlot.actions.some((action) => action === "Load" || action === "RetryLoad")) {
      actions.push("load")
    }
    if (detailIsLocal
      && detailIsSelected
      && primarySlot
      && primarySlot._tag === "ConfiguredLocal"
      && primarySlot.actions.includes("Stop")) actions.push("stop")
    if (detailCatalogCandidate) actions.push("catalog")
    return actions
  }, [detail, detailCatalogCandidate, detailIsLocal, detailIsSelected, hardware, primarySlot, residentAllocation])
  const detailActionCursor = useBoundedCursor(detailActions.length)
  const emptyActionCursor = useBoundedCursor(EMPTY_MODEL_ACTIONS.length)
  const focusedDetailAction = detailActions[detailActionCursor.index]

  const statusFor = useCallback((model: ProviderModelCatalogEntry): string => {
    const isSelected = providerModelKey(model) === selectedKey
    if (model.providerId === LOCAL_PROVIDER_ID) {
      const memoryConditions = providerModelMemoryConditions(model, hardware, residentAllocation)
      if (isSelected
        && primarySlot?._tag === "ConfiguredLocal"
        && Option.isSome(primarySlot.instance)
        && memoryConditions.lacksCurrentHeadroom) return "Selected"
      const memoryLabel = modelMemoryStatusLabel(memoryConditions)
      if (memoryLabel !== "") return memoryLabel
      if (model.availability._tag === "Disabled") return "Unavailable"
    }
    if (isSelected) return "Selected"
    return model.providerId === LOCAL_PROVIDER_ID ? "Installed" : "Available"
  }, [hardware, primarySlot, residentAllocation, selectedKey])

  const choose = useCallback((model: ProviderModelCatalogEntry) => {
    if (!model.supportedSlots.includes(PRIMARY_SLOT_ID)) return
    if (model.availability._tag !== "Available") return
    config.updateSlotModel(PRIMARY_SLOT_ID, model.providerId, model.providerModelId)
  }, [config])

  const toggleFavorite = useCallback((model: ProviderModelCatalogEntry) => {
    config.setModelFavorite({
      providerId: model.providerId,
      providerModelId: model.providerModelId,
    }, !currentFavoriteKeys.has(providerModelKey(model)))
  }, [config, currentFavoriteKeys])

  const runDetailAction = useCallback((action: typeof detailActions[number]) => {
    if (!detail) return
    if (action === "select") choose(detail)
    else if (action === "load") void slotActions.load(PRIMARY_SLOT_ID)
    else if (action === "stop" && primarySlot) {
      Option.match(modelSlotInstanceId(primarySlot), {
        onNone: () => {},
        onSome: slotActions.stop,
      })
    }
    else if (detailCatalogCandidate) openCatalogDetail(detailCatalogCandidate.configurationId)
  }, [choose, detail, detailCatalogCandidate, openCatalogDetail, primarySlot, slotActions])

  useKeyboard(useCallback((key: KeyEvent) => {
    if (key.defaultPrevented) return
    if (detail) {
      if (key.name === "f" && !key.ctrl && !key.meta && !key.option) {
        key.preventDefault()
        toggleFavorite(detail)
        return
      }
      if (key.name === "escape") {
        key.preventDefault()
        setDetailId(null)
        setRootSwitchingEnabled(true)
        return
      }
      if (key.name === "up" && detailActions.length > 0) {
        key.preventDefault()
        detailActionCursor.previous()
        return
      }
      if (key.name === "down" && detailActions.length > 0) {
        key.preventDefault()
        detailActionCursor.next()
        return
      }
      if ((key.name === "return" || key.name === "enter") && focusedDetailAction) {
        key.preventDefault()
        runDetailAction(focusedDetailAction)
      }
      return
    }
    if (eligible.length === 0) {
      if (key.name === "up" || key.name === "k") {
        key.preventDefault()
        emptyActionCursor.previous()
        return
      }
      if (key.name === "down" || key.name === "j") {
        key.preventDefault()
        emptyActionCursor.next()
        return
      }
      if (key.name === "return" || key.name === "enter") {
        key.preventDefault()
        openRoot(EMPTY_MODEL_ACTIONS[emptyActionCursor.index]!.root)
        return
      }
    }
    if ((key.name === "up" || key.name === "k") && eligible.length > 0) {
      key.preventDefault()
      setCursorId(providerModelKey(eligible[Math.max(0, cursorIndex - 1)]!))
      return
    }
    if ((key.name === "down" || key.name === "j") && eligible.length > 0) {
      key.preventDefault()
      setCursorId(providerModelKey(eligible[Math.min(eligible.length - 1, cursorIndex + 1)]!))
      return
    }
    if ((key.name === "return" || key.name === "enter") && cursor) {
      key.preventDefault()
      choose(cursor)
      return
    }
    if (key.name === "f" && !key.ctrl && !key.meta && !key.option && cursor) {
      key.preventDefault()
      toggleFavorite(cursor)
      return
    }
    if (key.name === "d" && cursor) {
      key.preventDefault()
      detailActionCursor.reset()
      setDetailId(providerModelKey(cursor))
      setRootSwitchingEnabled(false)
      return
    }
    if (key.name === "r") {
      key.preventDefault()
      config.refreshModels()
      return
    }
  }, [choose, config, cursor, cursorIndex, detail, detailActionCursor, detailActions.length, eligible, emptyActionCursor, focusedDetailAction, openRoot, runDetailAction, setRootSwitchingEnabled, toggleFavorite]))

  if (detail) {
    const detailActionLabel = {
      select: "Use this model",
      load: "Load model",
      stop: "Stop model",
      catalog: "View in Catalog",
    } as const
    return (
      <>
        <MenuHeader
          title="Models"
          selection={detail.displayName}
          onSectionClick={() => {
            setDetailId(null)
            setRootSwitchingEnabled(true)
          }}
          hints={detailActions.length > 0 ? "↑↓ navigate · Enter choose · F favorite · Esc back" : "F favorite · Esc back"}
        />
        <box style={{ flexGrow: 1, minHeight: 0, flexDirection: "column", paddingLeft: 2, paddingRight: 2, paddingTop: 1 }}>
          <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>
            {currentFavoriteKeys.has(providerModelKey(detail)) ? "★ " : ""}{detail.displayName}
          </text>
          <text style={{ fg: theme.muted }}>
            {detailIsLocal ? "Local" : "Cloud"} · {formatContextWindow(detail.contextWindow)} context · {statusFor(detail)}
          </text>
          {detailIsLocal && modelMemoryStatusDetail(providerModelMemoryConditions(detail, hardware, residentAllocation)) !== "" && (
            <text style={{ fg: theme.warning }}>
              {modelMemoryStatusDetail(providerModelMemoryConditions(detail, hardware, residentAllocation))}
            </text>
          )}
          <text style={{ fg: theme.muted }}>
            {detail.capabilities.vision ? "Vision" : "No vision"} · Tools · {detail.capabilities.reasoning.supported ? "Reasoning" : "No reasoning"}
          </text>
          <box style={{ paddingTop: 1, flexDirection: "column" }}>
            {detailIsSelected && <text style={{ fg: theme.success }}>● Current model</text>}
            {detailActions.map((action, index) => (
              <MenuAction
                key={action}
                label={detailActionLabel[action]}
                focused={index === detailActionCursor.index}
                tone={action === "select" ? "primary" : action === "catalog" ? "link" : "normal"}
                onClick={() => runDetailAction(action)}
                onMouseOver={() => detailActionCursor.select(index)}
              />
            ))}
          </box>
        </box>
      </>
    )
  }

  return (
    <>
      <MenuHeader
        title="Models"
        subtitle="Choose a model"
        summary={`${eligible.filter((model) => model.providerId === LOCAL_PROVIDER_ID).length} local · ${eligible.filter((model) => model.providerId !== LOCAL_PROVIDER_ID).length} cloud`}
        hints={eligible.length === 0
          ? "↑↓ choose · Enter open · R refresh · Esc close"
          : "↑↓ choose · Enter select · F favorite · D details · R refresh · Esc close"}
      />
      <scrollbox
        scrollX={false}
        style={{
          flexGrow: 1,
          minHeight: 0,
          rootOptions: { backgroundColor: theme.menuBg },
          wrapperOptions: { border: false, backgroundColor: theme.menuBg },
          viewportOptions: { backgroundColor: theme.menuBg },
          contentOptions: { flexDirection: "column", paddingLeft: 2, paddingRight: 2, paddingTop: 1 },
        }}
      >
        <box style={{ flexDirection: "row", width: "100%" }}>
          <text style={{ fg: theme.muted, width: 2 }}> </text>
          <text style={{ fg: theme.muted, width: 2 }}> </text>
          <text style={{ fg: theme.muted, flexGrow: 1 }}>MODEL</text>
          <text style={{ fg: theme.muted, width: 14 }}>REQUIREMENTS</text>
          <text style={{ fg: theme.muted, width: 9 }}>CONTEXT</text>
          <text style={{ fg: theme.muted, width: 23 }}>STATUS</text>
        </box>
        {assessing.map((model, index) => (
          <box
            key={`assessing:${model.targetId}`}
            style={{
              flexDirection: "row",
              width: "100%",
              backgroundColor: index % 2 === 0 ? theme.menuBg : theme.menuAltBg,
            }}
          >
            <text style={{ width: 2 }}> </text>
            <text style={{ width: 2 }}> </text>
            <text style={{ fg: theme.foreground, flexGrow: 1 }}>{model.displayName}</text>
            <text style={{ fg: theme.muted, width: 14 }}>{assessingRequirementFor(model)}</text>
            <text style={{ fg: theme.muted, width: 9 }}>{formatContextWindow(model.maximumContextLength)}</text>
            <text style={{ fg: theme.primary, width: 23 }}>Assessing</text>
          </box>
        ))}
        {eligible.length === 0 ? (
          <box style={{ flexDirection: "column", paddingLeft: 2 }}>
            <text style={{ fg: theme.warning, marginLeft: 2 }}>No model is currently available.</text>
            {EMPTY_MODEL_ACTIONS.map((action, index) => (
              <MenuAction
                key={action.root}
                label={action.label}
                focused={index === emptyActionCursor.index}
                onClick={() => openRoot(action.root)}
                onMouseOver={() => emptyActionCursor.select(index)}
              />
            ))}
          </box>
        ) : eligible.map((model, index) => {
          const focused = index === cursorIndex
          const active = providerModelKey(model) === selectedKey
          const favorite = currentFavoriteKeys.has(providerModelKey(model))
          const rowIndex = assessing.length + index
          return (
            <Button
              key={providerModelKey(model)}
              onClick={() => choose(model)}
              onMouseOver={() => setCursorId(providerModelKey(model))}
              style={{
                flexDirection: "row",
                width: "100%",
                backgroundColor: active
                  ? focused ? theme.foreground : theme.primary
                  : focused
                  ? theme.surfaceHover
                  : rowIndex % 2 === 0 ? theme.menuBg : theme.menuAltBg,
              }}
            >
              <text style={{ fg: active ? theme.menuBg : focused ? theme.primary : theme.foreground, width: 2 }}>{active ? "●" : focused ? "›" : " "}</text>
              <text style={{ fg: active ? theme.menuBg : theme.warning, width: 2 }}>{favorite ? "★" : " "}</text>
              <text style={{ fg: active ? theme.menuBg : focused ? theme.primary : theme.foreground, flexGrow: 1 }} attributes={active ? TextAttributes.BOLD : TextAttributes.NONE}>{model.displayName}</text>
              <text style={{ fg: active ? theme.menuBg : theme.muted, width: 14 }}>{requirementFor(model)}</text>
              <text style={{ fg: active ? theme.menuBg : theme.muted, width: 9 }}>{formatContextWindow(model.contextWindow)}</text>
              <text style={{ fg: active ? theme.menuBg : theme.muted, width: 23 }} attributes={active ? TextAttributes.BOLD : TextAttributes.NONE}>{statusFor(model)}</text>
            </Button>
          )
        })}
        {Result.isFailure(config.catalog) && (
          <text style={{ fg: theme.error }}>Unable to refresh the provider model catalog; showing the last usable state when available.</text>
        )}
        {Result.isFailure(config.slotUpdate) && (
          <text style={{ fg: theme.error }}>Failed to update model selection.</text>
        )}
        {Result.isFailure(config.favoriteUpdate) && (
          <text style={{ fg: theme.error }}>Failed to update model favorite.</text>
        )}
      </scrollbox>
    </>
  )
})

const recommendationLabel = (recommendation: Option.Option<LocalModelRecommendation>): string =>
  Option.match(recommendation, {
    onNone: () => "",
    onSome: ({ intent }) => ({
      balanced: "Balanced",
      best_quality: "Best quality",
      fastest: "Fastest",
      lightweight: "Lightweight",
    })[intent],
  })

const intelligenceLabel = ({ recommendationEvidence }: LocalModelCatalogCandidate): string =>
  Option.match(recommendationEvidence, {
    onNone: () => "—",
    onSome: ({ intelligence }) => Option.match(intelligence, {
      onNone: () => "—",
      onSome: ({ score }) => `${Math.round(score)}/100`,
    }),
  })

const qualityLabel = ({ recommendationEvidence }: LocalModelCatalogCandidate): string =>
  Option.match(recommendationEvidence, {
    onNone: () => "—",
    onSome: ({ fidelityRank }) => fidelityRank >= 75
      ? "Near original"
      : fidelityRank >= 55
        ? "Very high"
        : fidelityRank >= 45 ? "High" : "Reduced",
  })

const recommendationEvidenceLabel = (candidate: LocalModelCatalogCandidate): string =>
  Option.isNone(candidate.recommendationEvidence)
    ? "recommendation evidence unavailable"
    : `intelligence ${intelligenceLabel(candidate)} · ${qualityLabel(candidate)}`

const qualityEvidence = ({ recommendationEvidence }: LocalModelCatalogCandidate): readonly string[] =>
  Option.match(recommendationEvidence, {
    onNone: () => [],
    onSome: ({ qualityEvidence: evidence }) => evidence,
  })

const catalogStatus = (candidate: LocalModelCatalogCandidate): string => {
  if (candidate.download._tag === "NotDownloaded"
    || candidate.download._tag === "Cancelled") return "Available"
  if (candidate.download._tag === "Downloading") {
    return `Downloading ${Math.round(candidate.download.completedBytes / Math.max(1, candidate.download.totalBytes) * 100)}%`
  }
  if (candidate.download._tag === "Failed") return "Download failed"
  if (candidate.availability._tag === "Unavailable") return "Unavailable"
  return "Installed"
}

const CatalogMenu = memo(function CatalogMenu({
  initialCatalogDetailId,
  setRootSwitchingEnabled,
}: CatalogMenuProps) {
  const theme = useTheme()
  const localModels = useLocalModels()
  const modelActions = useLocalModelActions()
  const slotActions = useModelSlotActions()
  const snapshot = Result.value(localModels)
  const catalogCandidates = Option.match(snapshot, {
    onNone: () => [] as readonly LocalModelCatalogCandidate[],
    onSome: (models) =>
      models.recommendations._tag === "Ready" ? models.recommendations.catalog : [],
  })
  const recommendations = Option.match(snapshot, {
    onNone: () => [] as readonly LocalModelRecommendation[],
    onSome: (models) =>
      models.recommendations._tag === "Ready" ? models.recommendations.entries : [],
  })
  const recommendationsReady = Option.exists(
    snapshot,
    (models) => models.recommendations._tag === "Ready",
  )
  const recommendationFor = useCallback((candidate: LocalModelCatalogCandidate) =>
    Option.fromNullable(recommendations.find((recommendation) =>
      recommendation.candidate.configurationId === candidate.configurationId)), [recommendations])
  const candidates = [...catalogCandidates].sort((left, right) => {
    const leftInstalled = left.download._tag === "Downloaded"
    const rightInstalled = right.download._tag === "Downloaded"
    return leftInstalled === rightInstalled ? 0 : leftInstalled ? -1 : 1
  })
  const [cursorId, setCursorId] = useState<string | null>(null)
  const [detailId, setDetailId] = useState<string | null>(initialCatalogDetailId)
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null)
  const cursorIndex = Math.max(0, candidates.findIndex(({ configurationId }) =>
    configurationId === cursorId))
  const cursor = candidates[cursorIndex]
  const detail = candidates.find(({ configurationId }) => configurationId === detailId) ?? null
  const progress = Option.match(snapshot, {
    onNone: () => [],
    onSome: (models) => localInferenceProgressLines(models.recommendations.progress),
  })
  const runningProgress = progress.find((line) => line.state === "running")
  const spinner = useSpinnerFrame(runningProgress !== undefined)
  const detailActions = useMemo(() => {
    if (!detail) return [] as readonly ("primary" | "cancel" | "select")[]
    const actions: ("primary" | "cancel" | "select")[] = []
    if (detail.download._tag === "Downloading") actions.push("cancel")
    else if (detail.download._tag === "Downloaded") {
      if (detail.availability._tag === "Available") actions.push("select")
    }
    else actions.push("primary")
    return actions
  }, [detail])
  const detailActionCursor = useBoundedCursor(detailActions.length)
  const focusedDetailAction = detailActions[detailActionCursor.index]

  const primaryAction = useCallback((candidate: LocalModelCatalogCandidate) => {
    if (candidate.download._tag === "Downloading"
      || candidate.download._tag === "Downloaded") return
    void modelActions.download(candidate.targetId)
  }, [modelActions])

  const selectCandidate = useCallback((candidate: LocalModelCatalogCandidate) => {
    if (candidate.availability._tag !== "Available"
      || Result.isWaiting(modelActions.createOfferingResult)) return
    const providerModelId = Option.flatMap(snapshot, ({ models }) => Option.fromNullable(
      models.find(({ targetId }) => targetId === candidate.targetId)
        ?.offerings.find(({ configurationId }) => configurationId === candidate.configurationId)
        ?.providerModelId,
    ))
    const assign = (id: ProviderModelId) => slotActions.assign(PRIMARY_SLOT_ID, {
      providerId: LOCAL_PROVIDER_ID,
      providerModelId: id,
      reasoningEffort: Option.getOrElse(
        candidate.capabilities.reasoning.defaultEffort,
        () => ReasoningEffortSchema.make("none"),
      ),
    })
    if (Option.isSome(providerModelId)) {
      void assign(providerModelId.value)
      return
    }
    void modelActions.createOffering(candidate.configurationId).then(
      assign,
      () => undefined,
    )
  }, [modelActions, slotActions, snapshot])

  const runDetailAction = useCallback((action: typeof detailActions[number]) => {
    if (!detail) return
    if (action === "primary") {
      primaryAction(detail)
      return
    }
    if (action === "cancel") {
      if (detail.download._tag === "Downloading") {
        modelActions.cancel(detail.download.attemptIds)
      }
      return
    }
    if (action === "select") {
      selectCandidate(detail)
      return
    }
  }, [detail, modelActions, primaryAction, selectCandidate])

  useKeyboard(useCallback((key: KeyEvent) => {
    if (key.defaultPrevented) return
    if (detail) {
      if (key.name === "escape") {
        key.preventDefault()
        setDetailId(null)
        setRootSwitchingEnabled(true)
      } else if (key.name === "up" && detailActions.length > 0) {
        key.preventDefault()
        detailActionCursor.previous()
      } else if (key.name === "down" && detailActions.length > 0) {
        key.preventDefault()
        detailActionCursor.next()
      } else if ((key.name === "return" || key.name === "enter") && focusedDetailAction) {
        key.preventDefault()
        runDetailAction(focusedDetailAction)
      }
      return
    }
    if (pendingDeleteId !== null) {
      const confirmsDelete = key.name === "y"
        && !key.ctrl
        && !key.meta
        && !key.option
      if (confirmsDelete) {
        const candidate = candidates.find(({ configurationId }) => configurationId === pendingDeleteId)
        if (candidate?.download._tag === "Downloaded") modelActions.delete(candidate.targetId)
        setPendingDeleteId(null)
        key.preventDefault()
        return
      }
      setPendingDeleteId(null)
      if (key.name === "escape" || key.name === "backspace" || key.name === "y" || key.name === "n") {
        key.preventDefault()
        return
      }
    }
    if ((key.name === "up" || key.name === "k") && candidates.length > 0) {
      key.preventDefault()
      setCursorId(candidates[Math.max(0, cursorIndex - 1)]!.configurationId)
    } else if ((key.name === "down" || key.name === "j") && candidates.length > 0) {
      key.preventDefault()
      setCursorId(candidates[Math.min(candidates.length - 1, cursorIndex + 1)]!.configurationId)
    } else if ((key.name === "return" || key.name === "enter") && cursor) {
      key.preventDefault()
      detailActionCursor.reset()
      setDetailId(cursor.configurationId)
      setRootSwitchingEnabled(false)
    } else if (key.name === "d" && cursor) {
      key.preventDefault()
      primaryAction(cursor)
    } else if (key.name === "s" && cursor && cursor.availability._tag === "Available") {
      key.preventDefault()
      selectCandidate(cursor)
    } else if (key.name === "backspace" && cursor) {
      if (cursor.download._tag === "Downloading") {
        modelActions.cancel(cursor.download.attemptIds)
        key.preventDefault()
      } else if (cursor.download._tag === "Downloaded") {
        setPendingDeleteId(cursor.configurationId)
        key.preventDefault()
      }
    }
  }, [candidates, cursor, cursorIndex, detail, detailActionCursor, detailActions.length, focusedDetailAction, modelActions, pendingDeleteId, primaryAction, runDetailAction, selectCandidate, setRootSwitchingEnabled]))

  if (detail) {
    const recommendation = recommendationFor(detail)
    const downloading = detail.download._tag === "Downloading"
    const downloaded = detail.download._tag === "Downloaded"
    const failed = detail.download._tag === "Failed"
    const detailActionLabel = {
      primary: failed ? "Retry download" : "Download",
      cancel: "Cancel download",
      select: "Select this model",
    } as const
    return (
      <>
        <MenuHeader
          title="Catalog"
          selection={detail.displayName}
          onSectionClick={() => {
            setDetailId(null)
            setRootSwitchingEnabled(true)
          }}
          hints="↑↓ navigate · Enter choose · Esc back"
        />
        <scrollbox scrollX={false} style={{
          flexGrow: 1,
          minHeight: 0,
          rootOptions: { backgroundColor: theme.menuBg },
          wrapperOptions: { border: false, backgroundColor: theme.menuBg },
          viewportOptions: { backgroundColor: theme.menuBg },
          contentOptions: { flexDirection: "column", paddingLeft: 2, paddingRight: 2, paddingTop: 1 },
        }}>
          <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>{detail.displayName}</text>
          <text style={{ fg: theme.muted }}>{detail.description}</text>
          {Option.isSome(recommendation) && (
            <>
              <text style={{ fg: theme.primary }}>{recommendationLabel(recommendation)}</text>
              <text style={{ fg: theme.muted }}>{recommendation.value.explanation}</text>
            </>
          )}
          <text style={{ fg: theme.foreground, marginTop: 1 }} attributes={TextAttributes.BOLD}>Calibrated for this machine</text>
          <text style={{ fg: theme.muted }}>
            {formatBytes(requiredMemoryBytes(detail.memory))} memory · {detail.quantization} · {recommendationEvidenceLabel(detail)}
          </text>
          <text style={{ fg: theme.muted }}>
            {performanceRangeSpeedLabel(detail, "tokens/sec")}
          </text>
          <text style={{ fg: failed ? theme.error : downloading || downloaded ? theme.primary : theme.muted }}>
            {catalogStatus(detail)}
          </text>
          {failed && <text style={{ fg: theme.error }}>{detail.download.failure.message}</text>}
          {Result.isFailure(modelActions.createOfferingResult) && (
            <text style={{ fg: theme.error }}>Failed to create the local model offering.</text>
          )}
          <box style={{ paddingTop: 1, flexDirection: "column" }}>
            {detailActions.map((action, index) => (
              <MenuAction
                key={action}
                label={detailActionLabel[action]}
                focused={index === detailActionCursor.index}
                tone={action === "primary" || action === "select" ? "primary" : "warning"}
                onClick={() => runDetailAction(action)}
                onMouseOver={() => detailActionCursor.select(index)}
              />
            ))}
          </box>
          <text style={{ fg: theme.muted, marginTop: 1 }}>License: {detail.license}</text>
          {qualityEvidence(detail).map((evidence) => <text key={evidence} style={{ fg: theme.muted }}>{evidence}</text>)}
        </scrollbox>
      </>
    )
  }

  return (
    <>
      <MenuHeader
        title="Catalog"
        subtitle="Find and download local models"
        summary={`${candidates.length} compatible`}
        hints="↑↓ navigate · Enter details · D download · S select · Backspace cancel/remove · Esc close"
      />
      <scrollbox scrollX={false} style={{
        flexGrow: 1,
        minHeight: 0,
        rootOptions: { backgroundColor: theme.menuBg },
        wrapperOptions: { border: false, backgroundColor: theme.menuBg },
        viewportOptions: { backgroundColor: theme.menuBg },
        contentOptions: { flexDirection: "column", paddingLeft: 2, paddingRight: 2, paddingTop: 1 },
      }}>
        <box style={{ flexDirection: "row", width: "100%" }}>
          <text style={{ fg: theme.muted, width: 2 }}> </text>
          <text style={{ fg: theme.muted, flexGrow: 1 }}>MODEL</text>
          <text style={{ fg: theme.muted, width: 16 }}>RECOMMENDATION</text>
          <text style={{ fg: theme.muted, width: 12 }}>MEMORY</text>
          <text style={{ fg: theme.muted, width: 14 }}>INTELLIGENCE</text>
          <text style={{ fg: theme.muted, width: 14 }}>QUALITY</text>
          <text style={{ fg: theme.muted, width: 12 }}>SPEED</text>
          <text style={{ fg: theme.muted, width: 18 }}>STATUS</text>
        </box>
        {runningProgress && (
          <text style={{ fg: theme.primary, marginLeft: 2 }}>
            {spinner} {runningProgress.label}{runningProgress.metadata}
          </text>
        )}
        {candidates.length === 0 && recommendationsReady ? (
          <text style={{ fg: theme.warning, marginLeft: 2 }}>
            No compatible recommended models are currently available.
          </text>
        ) : candidates.map((candidate, index) => {
          const focused = index === cursorIndex
          const pendingDelete = pendingDeleteId === candidate.configurationId
          return (
            <Button
              key={candidate.configurationId}
              onClick={() => {
                setPendingDeleteId(null)
                detailActionCursor.reset()
                setDetailId(candidate.configurationId)
                setRootSwitchingEnabled(false)
              }}
              onMouseOver={() => {
                setCursorId(candidate.configurationId)
                if (pendingDeleteId !== candidate.configurationId) setPendingDeleteId(null)
              }}
              style={{
                flexDirection: "row",
                width: "100%",
                backgroundColor: focused
                  ? theme.surfaceHover
                  : index % 2 === 0 ? theme.menuBg : theme.menuAltBg,
              }}
            >
              <text style={{ fg: focused ? theme.primary : theme.foreground, width: 2 }}>{focused ? "›" : " "}</text>
              <text style={{ fg: focused ? theme.primary : theme.foreground, flexGrow: 1 }}>
                {candidate.displayName}<span style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>{` (${candidate.quantizationName})`}</span>
              </text>
              <text style={{ fg: theme.primary, width: 16 }}>{recommendationLabel(recommendationFor(candidate))}</text>
              <text style={{ fg: theme.muted, width: 12 }}>{formatBytes(requiredMemoryBytes(candidate.memory))}</text>
              <text style={{ fg: theme.muted, width: 14 }}>{intelligenceLabel(candidate)}</text>
              <text style={{ fg: theme.muted, width: 14 }}>{qualityLabel(candidate)}</text>
              <text style={{ fg: theme.muted, width: 12 }}>{performanceRangeSpeedLabel(candidate, "t/s")}</text>
              <text
                style={{
                  fg: pendingDelete
                    ? theme.warning
                    : candidate.download._tag === "Failed"
                      ? theme.error
                      : candidate.download._tag === "Downloading" || candidate.download._tag === "Downloaded"
                        ? theme.primary
                        : theme.muted,
                  width: 18,
                }}
              >
                {pendingDelete ? "Delete [y/n]" : catalogStatus(candidate)}
              </text>
            </Button>
          )
        })}
      </scrollbox>
    </>
  )
})

const HardwareMenu = memo(function HardwareMenu() {
  const theme = useTheme()
  const hardwareState = useLocalInferenceHardware()
  const config = useModelConfig()
  const slotActions = useModelSlotActions()
  const hardwareSnapshot = Result.value(hardwareState)
  const slotsSnapshot = Result.value(config.slots)
  const currentSlot = Option.flatMap(slotsSnapshot, ({ state }) => {
    const slot = state.slots.primary
    return slot._tag === "ConfiguredLocal"
      ? Option.some(slot)
      : Option.none()
  })
  const currentModel = deriveCurrentLocalModel(
    Option.map(currentSlot, (slot) => slot),
  )
  const currentResidentAllocation = Option.flatMap(
    currentSlot,
    modelSlotResidentAllocation,
  )
  const action = Option.match(currentSlot, {
    onNone: () => Option.none<"load" | "stop">(),
    onSome: (slot) => slot.actions.includes("Stop")
      ? Option.some("stop" as const)
      : slot.actions.some((candidate) => candidate === "Load" || candidate === "RetryLoad")
        ? Option.some("load" as const)
        : Option.none(),
  })
  const runAction = useCallback(() => {
    if (Option.isNone(action)) return
    if (action.value === "load") {
      void slotActions.load(PRIMARY_SLOT_ID)
      return
    }
    Option.flatMap(currentSlot, modelSlotInstanceId).pipe(
      Option.match({
        onNone: () => {},
        onSome: slotActions.stop,
      }),
    )
  }, [action, currentSlot, slotActions])

  useKeyboard(useCallback((key: KeyEvent) => {
    if (key.defaultPrevented) return
    if (key.name === "up" || key.name === "down") {
      key.preventDefault()
      return
    }
    if ((key.name === "return" || key.name === "enter") && Option.isSome(action)) {
      key.preventDefault()
      runAction()
    }
  }, [action, runAction]))

  return (
    <>
      <MenuHeader title="Hardware" hints="↑↓ navigate · Enter choose · Esc close" />
      <scrollbox
        scrollX={false}
        style={{
          flexGrow: 1,
          minHeight: 0,
          rootOptions: { backgroundColor: theme.menuBg },
          wrapperOptions: { border: false, backgroundColor: theme.menuBg },
          viewportOptions: { backgroundColor: theme.menuBg },
          contentOptions: { flexDirection: "column", paddingLeft: 2, paddingRight: 2, paddingTop: 1 },
        }}
      >
        {Option.match(hardwareSnapshot, {
          onNone: () => (
            <text style={{ fg: Result.isFailure(hardwareState) ? theme.error : theme.muted }}>
              {Result.isFailure(hardwareState) ? "Hardware detection is unavailable." : "Detecting local-inference hardware…"}
            </text>
          ),
          onSome: (detectedHardware) => {
            const hardware = describeLocalHardware(detectedHardware)
            return (
              <>
                <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>{hardware.system.name}</text>
                {hardware.system.details.map((line) => <text key={line} style={{ fg: theme.muted }}>{line}</text>)}
                {hardware.accelerators.map((accelerator) => (
                  <text key={`${accelerator.name}:${accelerator.details}`} style={{ fg: theme.muted }}>{accelerator.name} · {accelerator.details}</text>
                ))}
                {hardware.accelerators.length === 0 && !detectedHardware.memoryDomains.some((domain) => domain.kind === "UnifiedMemory") && (
                  <text style={{ fg: theme.muted }}>CPU inference · No GPU detected</text>
                )}
              </>
            )
          },
        })}
        <box style={{ flexDirection: "column", marginTop: 1 }}>
          <text style={{ fg: theme.muted }} attributes={TextAttributes.BOLD}>CURRENT MODEL</text>
          {currentModel._tag === "NoSelection"
            ? <text style={{ fg: theme.muted }}>No local model selected</text>
            : (() => {
              const actualAllocation = currentModel._tag === "Running"
                ? Option.some(currentModel.allocation)
                : currentModel._tag === "Loading" || currentModel._tag === "Stopping"
                  ? currentModel.allocation
                  : Option.none()
              const status = currentModel._tag === "NotLoaded"
                ? "NOT LOADED"
                : currentModel._tag === "Loading"
                  ? `LOADING · ${currentModel.percentage}%`
                  : currentModel._tag === "Running"
                    ? "RUNNING"
                    : currentModel._tag === "Stopping"
                      ? "STOPPING"
                      : "FAILED"
              return (
                <>
                  <box style={{ flexDirection: "row" }}>
                    <text style={{ fg: theme.foreground, flexGrow: 1 }} attributes={TextAttributes.BOLD}>{currentModel.displayName}</text>
                    <text style={{ fg: currentModel._tag === "Running" ? theme.primary : theme.muted }}>{status}</text>
                  </box>
                  {currentModel._tag === "NotLoaded" || currentModel._tag === "Failed"
                    ? <ModelLoadPlanDetails />
                    : (
                        <box style={{ flexDirection: "row" }}>
                          <text style={{ fg: theme.muted, width: 20 }}>Context window</text>
                          <text style={{ fg: theme.foreground, width: 16 }}>
                            {Option.match(currentModel.contextWindow, {
                              onNone: () => "—",
                              onSome: (tokens) => `${formatContextWindow(tokens)} tokens`,
                            })}
                          </text>
                          <text style={{ fg: theme.muted, width: 16 }}>Parallelism</text>
                          <text style={{ fg: theme.foreground }}>
                            {Option.match(actualAllocation, {
                              onNone: () => "—",
                              onSome: (allocation) => String(allocation.parallelSequences),
                            })}
                          </text>
                        </box>
                      )}
                </>
              )
            })()}
        </box>
        {Option.match(hardwareSnapshot, {
          onNone: () => null,
          onSome: (state) => (
            <box style={{ flexDirection: "column", marginTop: 1 }}>
              {deriveHardwareMemoryView(state, currentResidentAllocation).domains.map((domain) =>
                <HardwareMemoryDomain key={domain.id} domain={domain} />)}
            </box>
          ),
        })}
        {Option.isSome(currentSlot) && (
          <box style={{ flexDirection: "column", marginTop: 1 }}>
            <text style={{ fg: theme.muted }} attributes={TextAttributes.BOLD}>ACTIONS</text>
            {Option.match(action, {
              onNone: () =>
                Option.exists(currentSlot.value.instance, (instance) =>
                  instance.lifecycle._tag === "Stopping")
                  ? <text style={{ fg: theme.muted }}>Stopping model…</text>
                  : currentSlot.value.availability._tag === "Unavailable"
                    ? <text style={{ fg: theme.muted }}>Unable to load model</text>
                    : <text style={{ fg: theme.muted }}>{"  "}Load model</text>,
              onSome: (currentAction) => (
                <MenuAction
                  label={currentAction === "load" ? "Load model" : "Stop model"}
                  focused
                  tone={currentAction === "load" ? "primary" : "normal"}
                  onClick={runAction}
                  onMouseOver={() => {}}
                />
              ),
            })}
          </box>
        )}
      </scrollbox>
    </>
  )
})

const ModelLoadPlanDetails = memo(function ModelLoadPlanDetails() {
  const theme = useTheme()
  const preview = usePreviewModelLoad(PRIMARY_SLOT_ID)
  const plan = Result.value(preview)
  return (
    <box style={{ flexDirection: "row" }}>
      <text style={{ fg: theme.muted, width: 20 }}>Context window</text>
      <text style={{ fg: theme.foreground, width: 16 }}>
        {Option.match(plan, {
          onNone: () => "—",
          onSome: ({ contextWindowTokens }) =>
            `${formatContextWindow(contextWindowTokens)} tokens`,
        })}
      </text>
      <text style={{ fg: theme.muted, width: 16 }}>Parallelism</text>
      <text style={{ fg: theme.foreground }}>
        {Option.match(plan, {
          onNone: () => Result.isFailure(preview) ? "Unable to load now" : "—",
          onSome: ({ parallelSequences }) => `${parallelSequences} if loaded now`,
        })}
      </text>
    </box>
  )
})

const CloudMenu = memo(function CloudMenu({
  setRootSwitchingEnabled,
}: CloudMenuProps) {
  const theme = useTheme()
  const platform = usePlatform()
  const settings = useSettingsState()
  const config = useModelConfig()
  const authSource = useAtomValue(authSourceAtom)
  const [mode, setMode] = useState<"root" | "edit" | "disconnect">("root")
  const [keyValue, setKeyValue] = useState("")
  const [validationError, setValidationError] = useState<string | null>(null)
  const auth = useMemo(() => deriveSettingsAuthInfo({
    apiKey: settings.apiKey,
    authSource,
    save: settings.saveApiKey,
    clear: settings.disconnectApiKey,
    saving: settings.saving,
    error: settings.saveError,
  }), [authSource, settings.apiKey, settings.disconnectApiKey, settings.saveApiKey, settings.saveError, settings.saving])
  const connected = auth.source !== "none"
  const cloudModels = catalogModels(config).filter((model) =>
    model.providerId !== LOCAL_PROVIDER_ID
    && model.availability._tag === "Available"
    && model.supportedSlots.includes(PRIMARY_SLOT_ID))
  const actionIds = useMemo<readonly CloudActionId[]>(() => auth.source === "none"
    ? ["add", "link"]
    : auth.source === "config"
      ? ["update", "disconnect", "link"]
      : ["link"], [auth.source])
  const actionCursor = useBoundedCursor(actionIds.length)
  const disconnectCursor = useBoundedCursor(2)
  const selectedAction = actionIds[actionCursor.index]

  const save = useCallback(() => {
    const trimmed = keyValue.trim()
    if (!trimmed) {
      setValidationError("API key is required")
      return
    }
    setValidationError(null)
    auth.save(trimmed)
  }, [auth, keyValue])

  const runAction = useCallback((action: CloudActionId) => {
    if (action === "add" || action === "update") {
      setMode("edit")
      setRootSwitchingEnabled(false)
      return
    }
    if (action === "disconnect") {
      disconnectCursor.reset()
      setMode("disconnect")
      setRootSwitchingEnabled(false)
      return
    }
    void platform.openLink(MAGNITUDE_CLOUD_URL)
  }, [disconnectCursor, platform, setRootSwitchingEnabled])

  useKeyboard(useCallback((key: KeyEvent) => {
    if (key.defaultPrevented) return
    if (mode === "edit") {
      if (key.name === "escape") {
        key.preventDefault()
        setMode("root")
        setRootSwitchingEnabled(true)
        return
      }
      if ((key.name === "return" || key.name === "enter") && !key.shift) {
        key.preventDefault()
        save()
      }
      return
    }
    if (mode === "disconnect") {
      if (key.name === "escape") {
        key.preventDefault()
        setMode("root")
        setRootSwitchingEnabled(true)
        return
      }
      if (key.name === "up") {
        key.preventDefault()
        disconnectCursor.previous()
        return
      }
      if (key.name === "down") {
        key.preventDefault()
        disconnectCursor.next()
        return
      }
      if (key.name === "return" || key.name === "enter") {
        key.preventDefault()
        if (disconnectCursor.index === 1) auth.clear()
        setMode("root")
        setRootSwitchingEnabled(true)
      }
      return
    }
    if (key.name === "up" && actionIds.length > 0) {
      key.preventDefault()
      actionCursor.previous()
      return
    }
    if (key.name === "down" && actionIds.length > 0) {
      key.preventDefault()
      actionCursor.next()
      return
    }
    if ((key.name === "return" || key.name === "enter") && selectedAction) {
      key.preventDefault()
      runAction(selectedAction)
    }
  }, [actionCursor, actionIds.length, auth, disconnectCursor, mode, runAction, save, selectedAction, setRootSwitchingEnabled]))

  if (mode === "edit") {
    const error = validationError ?? auth.error
    return (
      <>
        <MenuHeader
          title="Cloud"
          selection={connected ? "Update API key" : "Add API key"}
          onSectionClick={() => {
            setMode("root")
            setRootSwitchingEnabled(true)
          }}
          hints="Enter save · Esc cancel"
        />
        <box style={{ flexGrow: 1, flexDirection: "column", paddingLeft: 2, paddingRight: 2, paddingTop: 1 }}>
          <text style={{ fg: theme.foreground }}>API key</text>
          <box style={{ borderStyle: "single", borderColor: error ? theme.error : theme.primary, paddingLeft: 1, paddingRight: 1, width: "100%" }}>
            <SingleLineInput
              value={keyValue}
              onChange={(value) => {
                setKeyValue(value)
                setValidationError(null)
              }}
              placeholder="Paste Magnitude Cloud API key"
              focused
            />
          </box>
          {error && <text style={{ fg: theme.error }}>{error}</text>}
          <text style={{ fg: theme.muted }}>{auth.saving ? "Saving…" : "Enter to save"}</text>
        </box>
      </>
    )
  }

  if (mode === "disconnect") {
    return (
      <>
        <MenuHeader
          title="Cloud"
          selection="Disconnect"
          onSectionClick={() => {
            setMode("root")
            setRootSwitchingEnabled(true)
          }}
          hints="↑↓ navigate · Enter choose · Esc back"
        />
        <box style={{ flexGrow: 1, flexDirection: "column", paddingLeft: 2, paddingRight: 2, paddingTop: 1 }}>
          <text style={{ fg: theme.foreground }}>Disconnect Magnitude Cloud?</text>
          <text style={{ fg: theme.muted }}>Cloud models will no longer be available in Models.</text>
          <box style={{ paddingTop: 1, flexDirection: "column" }}>
            <MenuAction
              label="Cancel"
              focused={disconnectCursor.index === 0}
              onClick={() => {
                setMode("root")
                setRootSwitchingEnabled(true)
              }}
              onMouseOver={() => disconnectCursor.select(0)}
            />
            <MenuAction
              label="Disconnect"
              focused={disconnectCursor.index === 1}
              tone="error"
              onClick={() => {
                auth.clear()
                setMode("root")
                setRootSwitchingEnabled(true)
              }}
              onMouseOver={() => disconnectCursor.select(1)}
            />
          </box>
        </box>
      </>
    )
  }

  return (
    <>
      <MenuHeader title="Cloud" subtitle="Manage Magnitude Cloud connection" summary={connected ? "Connected" : "Not connected"} hints="↑↓ navigate" />
      <box style={{ flexGrow: 1, flexDirection: "column", paddingLeft: 2, paddingRight: 2, paddingTop: 1 }}>
        {auth.source === "none" && (
          <text style={{ fg: theme.muted }}>Magnitude Cloud provides hosted models and hosted research features.</text>
        )}
        {auth.source === "config" && (
          <text style={{ fg: theme.success }}>● Connected via API key {auth.maskedKey ? `(${auth.maskedKey})` : ""}</text>
        )}
        {auth.source === "env" && (
          <>
            <text style={{ fg: theme.success }}>● Connected via {auth.envVarName}</text>
            <text style={{ fg: theme.muted }}>This key is managed by the environment. Update it and relaunch to change it.</text>
          </>
        )}
        <box style={{ flexDirection: "column", paddingTop: 1 }}>
          {auth.source === "none" && (
            <Button
              onClick={() => runAction("add")}
              onMouseOver={() => actionCursor.select(actionIds.indexOf("add"))}
            >
              <text style={{ fg: theme.primary }}>{selectedAction === "add" ? "› " : "  "}Add API key</text>
            </Button>
          )}
          {auth.source === "config" && (
            <>
              <Button
                onClick={() => runAction("update")}
                onMouseOver={() => actionCursor.select(actionIds.indexOf("update"))}
              >
                <text style={{ fg: selectedAction === "update" ? theme.primary : theme.foreground }}>
                  {selectedAction === "update" ? "› " : "  "}Update API key
                </text>
              </Button>
              <Button
                onClick={() => runAction("disconnect")}
                onMouseOver={() => actionCursor.select(actionIds.indexOf("disconnect"))}
              >
                <text style={{ fg: selectedAction === "disconnect" ? theme.primary : theme.foreground }}>
                  {selectedAction === "disconnect" ? "› " : "  "}Disconnect
                </text>
              </Button>
            </>
          )}
          <box style={{ flexDirection: "row" }}>
            <text style={{ fg: theme.primary }}>{selectedAction === "link" ? "› " : "  "}</text>
            <Button
              onClick={() => runAction("link")}
              onMouseOver={() => actionCursor.select(actionIds.indexOf("link"))}
            >
              <text style={{ fg: theme.foreground }}>
                View dashboard{" "}
                <span
                  style={{ fg: selectedAction === "link" ? theme.link : theme.primary }}
                  attributes={TextAttributes.UNDERLINE}
                >
                  {MAGNITUDE_CLOUD_URL}↗
                </span>
              </text>
            </Button>
          </box>
        </box>
        {auth.error && <text style={{ fg: theme.error }}>{auth.error}</text>}
        {connected && cloudModels.length > 0 && (
          <box style={{ flexDirection: "column", paddingTop: 1 }}>
            <text style={{ fg: theme.muted }}>AVAILABLE MODELS</text>
            {cloudModels.map((model) => (
              <text key={providerModelKey(model)} style={{ fg: theme.foreground }}>
                {model.displayName}<span style={{ fg: theme.muted }}> · {formatContextWindow(model.contextWindow)} context</span>
              </text>
            ))}
          </box>
        )}
      </box>
    </>
  )
})
