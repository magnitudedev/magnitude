import {
  memo,
  useCallback,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react"
import { TextAttributes, type KeyEvent } from "@opentui/core"
import { useKeyboard } from "@opentui/react"
import { Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Cause, Option } from "effect"
import {
  deriveHardwareMemoryView,
  selectedSlotModel,
  usePlatform,
  useLocalInferenceHardware,
  useLocalInferenceState,
  useModelConfig,
  useSettingsState,
} from "@magnitudedev/client-common"
import {
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelCatalogLifecycle,
  type LocalModel,
  type LocalModelCatalogCandidate,
  type LocalModelRecommendation,
  type ProviderModelCatalogEntry,
} from "@magnitudedev/sdk"
import { Button } from "../../components/button"
import { HardwareMemoryDomain } from "../../components/hardware-memory-domain"
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

interface MenuRootProps {
  readonly openRoot: (root: ModelMenuRoot) => void
  readonly openCatalogDetail: (candidateId: string) => void
  readonly initialCatalogDetailId: string | null
  readonly setRootSwitchingEnabled: (enabled: boolean) => void
  readonly bindMenuKeyHandler: (handler: (key: KeyEvent) => void) => void
}

const nextRoot = (root: ModelMenuRoot, direction: -1 | 1): ModelMenuRoot => {
  const index = ROOTS.indexOf(root)
  return ROOTS[(index + direction + ROOTS.length) % ROOTS.length]!
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
  const [rootSwitchingEnabled, setRootSwitchingEnabled] = useState(true)
  const [hoveredRoot, setHoveredRoot] = useState<ModelMenuRoot | null>(null)
  const [catalogDetailId, setCatalogDetailId] = useState<string | null>(null)
  const menuKeyHandlerRef = useRef<(key: KeyEvent) => void>(() => {})
  const bindMenuKeyHandler = useCallback((handler: (key: KeyEvent) => void) => {
    menuKeyHandlerRef.current = handler
  }, [])

  const openRoot = useCallback((root: ModelMenuRoot) => {
    setCatalogDetailId(null)
    setRootSwitchingEnabled(true)
    setMenu({ open: true, root })
  }, [setMenu])
  const openCatalogDetail = useCallback((candidateId: string) => {
    setCatalogDetailId(candidateId)
    setRootSwitchingEnabled(false)
    setMenu({ open: true, root: "catalog" })
  }, [setMenu])
  const close = useCallback(() => {
    setMenu((current) => ({ ...current, open: false }))
  }, [setMenu])

  useKeyboard(useCallback((key: KeyEvent) => {
    if (!menu.open || key.defaultPrevented) return
    menuKeyHandlerRef.current(key)
    if (key.defaultPrevented) return
    if ((key.name === "left" || key.name === "right")
      && !key.ctrl && !key.meta && !key.option) {
      key.preventDefault()
      openRoot(nextRoot(menu.root, key.name === "left" ? -1 : 1))
      return
    }
    if (key.name === "tab" && !key.ctrl && !key.meta && !key.option) {
      key.preventDefault()
      openRoot(nextRoot(menu.root, key.shift ? -1 : 1))
      return
    }
    if (key.name === "escape") {
      key.preventDefault()
      close()
    }
  }, [close, menu.open, menu.root, openRoot, rootSwitchingEnabled]))

  if (!menu.open) return null

  return (
    <box
      style={{
        height: MENU_HEIGHT,
        maxHeight: "100%",
        minHeight: 0,
        width: "100%",
        marginTop: 1,
        flexShrink: 0,
        flexDirection: "column",
        backgroundColor: "transparent",
      }}
    >
      <box style={{ flexGrow: 1, minHeight: 0, flexDirection: "column", backgroundColor: theme.menuBg }}>
        {menu.root === "models" && <ModelsMenu openRoot={openRoot} openCatalogDetail={openCatalogDetail} initialCatalogDetailId={catalogDetailId} setRootSwitchingEnabled={setRootSwitchingEnabled} bindMenuKeyHandler={bindMenuKeyHandler} />}
        {menu.root === "catalog" && <CatalogMenu openRoot={openRoot} openCatalogDetail={openCatalogDetail} initialCatalogDetailId={catalogDetailId} setRootSwitchingEnabled={setRootSwitchingEnabled} bindMenuKeyHandler={bindMenuKeyHandler} />}
        {menu.root === "hardware" && <HardwareMenu openRoot={openRoot} openCatalogDetail={openCatalogDetail} initialCatalogDetailId={catalogDetailId} setRootSwitchingEnabled={setRootSwitchingEnabled} bindMenuKeyHandler={bindMenuKeyHandler} />}
        {menu.root === "cloud" && <CloudMenu openRoot={openRoot} openCatalogDetail={openCatalogDetail} initialCatalogDetailId={catalogDetailId} setRootSwitchingEnabled={setRootSwitchingEnabled} bindMenuKeyHandler={bindMenuKeyHandler} />}
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
          {rootSwitchingEnabled ? "←/→ switch menus" : "←/→ switch menus · Esc back"}
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
  bindMenuKeyHandler,
}: MenuRootProps) {
  const theme = useTheme()
  const config = useModelConfig()
  const local = useLocalInferenceState()
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
      && (model.availability._tag === "Available" || providerModelKey(model) === selectedKey))
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
  const [detailActionIndex, setDetailActionIndex] = useState(0)
  const cursorIndex = Math.max(0, eligible.findIndex((model) => providerModelKey(model) === cursorId))
  const cursor = eligible[cursorIndex]
  const detail = eligible.find((model) => providerModelKey(model) === detailId) ?? null
  const localSnapshot = Result.value(local.state)
  const localCatalogCandidates = Option.match(localSnapshot, {
    onNone: () => [] as readonly LocalModelCatalogCandidate[],
    onSome: ({ models: localModels }) =>
      localModels.recommendations._tag === "Ready" ? localModels.recommendations.catalog : [],
  })
  const requirementFor = (model: ProviderModelCatalogEntry): string => {
    if (model.providerId !== LOCAL_PROVIDER_ID) return "Cloud"
    return Option.match(model.runtimeMemoryBytes, {
      onNone: () => "—",
      onSome: formatBytes,
    })
  }
  const calibratingRequirementFor = (model: LocalModel): string => {
    const candidate = localCatalogCandidates.find(({ id }) =>
      model.catalogCandidateIds.includes(id))
    return candidate ? formatBytes(candidate.runtimeMemoryBytes) : "—"
  }
  const calibrating = Option.match(localSnapshot, {
    onNone: () => [] as readonly LocalModel[],
    onSome: ({ models: localModels }) => localModels.models.filter((model) =>
      model.download._tag === "Downloaded" && model.preparation._tag === "Preparing"),
  })
  const primarySlot = Option.match(localSnapshot, {
    onNone: () => null,
    onSome: (snapshot) => snapshot.slots.slots.primary,
  })
  const detailIsLocal = detail?.providerId === LOCAL_PROVIDER_ID
  const detailIsSelected = detail !== null && providerModelKey(detail) === selectedKey
  const detailLocalModel = detailIsLocal && detail
    ? Option.match(localSnapshot, {
        onNone: () => undefined,
        onSome: ({ models: localModels }) => localModels.models.find(({ preparation }) =>
          (preparation._tag === "Available" || preparation._tag === "Unavailable")
          && preparation.providerModelIds.includes(detail.providerModelId)),
      })
    : undefined
  const detailCatalogCandidate = detailLocalModel
    ? localCatalogCandidates.find(({ id }) => detailLocalModel.catalogCandidateIds.includes(id))
    : undefined
  const detailActions = useMemo(() => {
    if (!detail) return [] as readonly ("select" | "load" | "unload" | "catalog")[]
    const actions: ("select" | "load" | "unload" | "catalog")[] = []
    if (!detailIsSelected
      && detail.availability._tag === "Available"
      && detail.supportedSlots.includes(PRIMARY_SLOT_ID)) actions.push("select")
    if (detailIsLocal && detailIsSelected && primarySlot?._tag === "UnloadedLocalModel") actions.push("load")
    if (detailIsLocal && detailIsSelected && primarySlot?._tag === "Ready") actions.push("unload")
    if (detailCatalogCandidate) actions.push("catalog")
    return actions
  }, [detail, detailCatalogCandidate, detailIsLocal, detailIsSelected, primarySlot])
  const focusedDetailAction = detailActions[Math.min(detailActionIndex, Math.max(0, detailActions.length - 1))]

  const statusFor = useCallback((model: ProviderModelCatalogEntry): string => {
    const isSelected = providerModelKey(model) === selectedKey
    if (model.availability._tag === "Disabled") {
      return "Unavailable"
    }
    if (isSelected) return "Selected"
    return model.providerId === LOCAL_PROVIDER_ID ? "Installed" : "Available"
  }, [selectedKey])

  const choose = useCallback((model: ProviderModelCatalogEntry) => {
    if (!model.supportedSlots.includes(PRIMARY_SLOT_ID)) return
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
    else if (action === "load") local.loadModel(PRIMARY_SLOT_ID)
    else if (action === "unload") local.unloadModel(PRIMARY_SLOT_ID)
    else if (detailCatalogCandidate) openCatalogDetail(detailCatalogCandidate.id)
  }, [choose, detail, detailCatalogCandidate, local, openCatalogDetail])

  bindMenuKeyHandler(useCallback((key: KeyEvent) => {
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
        setDetailActionIndex((current) => Math.max(0, Math.min(detailActions.length - 1, current - 1)))
        return
      }
      if (key.name === "down" && detailActions.length > 0) {
        key.preventDefault()
        setDetailActionIndex((current) => Math.min(detailActions.length - 1, current + 1))
        return
      }
      if ((key.name === "return" || key.name === "enter") && focusedDetailAction) {
        key.preventDefault()
        runDetailAction(focusedDetailAction)
      }
      return
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
      setDetailActionIndex(0)
      setDetailId(providerModelKey(cursor))
      setRootSwitchingEnabled(false)
      return
    }
    if (key.name === "r") {
      key.preventDefault()
      config.refreshModels()
      return
    }
  }, [config, cursor, cursorIndex, detail, detailActions.length, eligible, focusedDetailAction, runDetailAction, setRootSwitchingEnabled, toggleFavorite]))

  if (detail) {
    const detailActionLabel = {
      select: "Use this model",
      load: "Load model",
      unload: "Unload model",
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
          <text style={{ fg: theme.muted }}>
            {detail.capabilities.vision ? "Vision" : "No vision"} · Tools · {detail.capabilities.reasoning.supported ? "Reasoning" : "No reasoning"}
          </text>
          <box style={{ paddingTop: 1, flexDirection: "column" }}>
            {detailIsSelected && <text style={{ fg: theme.success }}>● Current model</text>}
            {detailActions.map((action, index) => (
              <MenuAction
                key={action}
                label={detailActionLabel[action]}
                focused={index === Math.min(detailActionIndex, Math.max(0, detailActions.length - 1))}
                tone={action === "select" ? "primary" : action === "catalog" ? "link" : "normal"}
                onClick={() => runDetailAction(action)}
                onMouseOver={() => setDetailActionIndex(index)}
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
        hints="↑↓ choose · Enter select · F favorite · D details · R refresh · Esc close"
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
        {calibrating.map((model, index) => (
          <box
            key={`calibrating:${model.id}`}
            style={{
              flexDirection: "row",
              width: "100%",
              backgroundColor: index % 2 === 0 ? theme.menuBg : theme.menuAltBg,
            }}
          >
            <text style={{ width: 2 }}> </text>
            <text style={{ width: 2 }}> </text>
            <text style={{ fg: theme.foreground, flexGrow: 1 }}>{model.displayName}</text>
            <text style={{ fg: theme.muted, width: 14 }}>{calibratingRequirementFor(model)}</text>
            <text style={{ fg: theme.muted, width: 9 }}>{formatContextWindow(model.maximumContextLength)}</text>
            <text style={{ fg: theme.primary, width: 23 }}>Calibrating</text>
          </box>
        ))}
        {eligible.length === 0 ? (
          <box style={{ flexDirection: "column" }}>
            <text style={{ fg: theme.warning }}>No model is currently available.</text>
            <Button onClick={() => openRoot("cloud")}><text style={{ fg: theme.link }}>› Connect cloud models →</text></Button>
            <Button onClick={() => openRoot("catalog")}><text style={{ fg: theme.link }}>  Find a local model →</text></Button>
          </box>
        ) : eligible.map((model, index) => {
          const focused = index === cursorIndex
          const active = providerModelKey(model) === selectedKey
          const favorite = currentFavoriteKeys.has(providerModelKey(model))
          const rowIndex = calibrating.length + index
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

const qualityLabel = ({ fidelityRank }: LocalModelCatalogCandidate): string =>
  fidelityRank >= 75 ? "Near original" : fidelityRank >= 55 ? "Very high" : fidelityRank >= 45 ? "High" : "Reduced"

const catalogStatus = (candidate: LocalModelCatalogCandidate): string => {
  if (candidate.download._tag === "NotDownloaded") return "Available"
  if (candidate.download._tag === "Downloading") {
    return `Downloading ${Math.round(candidate.download.completedBytes / Math.max(1, candidate.download.totalBytes) * 100)}%`
  }
  if (candidate.download._tag === "Failed") return "Download failed"
  if (candidate.preparation._tag === "Calibrating") return "Calibrating"
  if (candidate.preparation._tag === "Unavailable") return "Unavailable"
  return "Installed"
}

const CatalogMenu = memo(function CatalogMenu({
  openRoot,
  initialCatalogDetailId,
  setRootSwitchingEnabled,
  bindMenuKeyHandler,
}: MenuRootProps) {
  const theme = useTheme()
  const local = useLocalInferenceState()
  const snapshot = Result.value(local.state)
  const catalogCandidates = Option.match(snapshot, {
    onNone: () => [] as readonly LocalModelCatalogCandidate[],
    onSome: ({ models }) =>
      models.recommendations._tag === "Ready" ? models.recommendations.catalog : [],
  })
  const recommendations = Option.match(snapshot, {
    onNone: () => [] as readonly LocalModelRecommendation[],
    onSome: ({ models }) =>
      models.recommendations._tag === "Ready" ? models.recommendations.entries : [],
  })
  const recommendationFor = useCallback((candidate: LocalModelCatalogCandidate) =>
    Option.fromNullable(recommendations.find((recommendation) =>
      recommendation.candidate.id === candidate.id)), [recommendations])
  const candidates = [...catalogCandidates].sort((left, right) => {
    const leftInstalled = left.download._tag === "Downloaded"
    const rightInstalled = right.download._tag === "Downloaded"
    return leftInstalled === rightInstalled ? 0 : leftInstalled ? -1 : 1
  })
  const [cursorId, setCursorId] = useState<string | null>(null)
  const [detailId, setDetailId] = useState<string | null>(initialCatalogDetailId)
  const [detailActionIndex, setDetailActionIndex] = useState(0)
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null)
  const cursorIndex = Math.max(0, candidates.findIndex(({ id }) => id === cursorId))
  const cursor = candidates[cursorIndex]
  const detail = candidates.find(({ id }) => id === detailId) ?? null
  const progress = Option.match(snapshot, {
    onNone: () => [],
    onSome: (state) => localInferenceProgressLines(state.models.recommendations.progress, Date.now()),
  })
  const runningProgress = progress.find((line) => line.state === "running")
  const detailActions = useMemo(() => {
    if (!detail) return [] as readonly ("primary" | "cancel" | "select")[]
    const actions: ("primary" | "cancel" | "select")[] = []
    if (detail.download._tag === "Downloading") actions.push("cancel")
    else if (detail.download._tag === "Downloaded") {
      if (detail.preparation._tag === "Installed") actions.push("select")
    }
    else actions.push("primary")
    return actions
  }, [detail])
  const focusedDetailAction = detailActions[Math.min(detailActionIndex, Math.max(0, detailActions.length - 1))]

  const primaryAction = useCallback((candidate: LocalModelCatalogCandidate) => {
    if (candidate.download._tag === "Downloading"
      || candidate.download._tag === "Downloaded") return
    local.downloadCatalogModel(candidate.id)
  }, [local])

  const selectCandidate = useCallback((candidate: LocalModelCatalogCandidate) => {
    if (candidate.preparation._tag !== "Installed") return
    local.selectCatalogModel(candidate.id)
  }, [local])

  const runDetailAction = useCallback((action: typeof detailActions[number]) => {
    if (!detail) return
    if (action === "primary") {
      primaryAction(detail)
      return
    }
    if (action === "cancel") {
      local.cancelCatalogModelDownload(detail.id)
      return
    }
    if (action === "select") {
      selectCandidate(detail)
      return
    }
  }, [detail, local, primaryAction, selectCandidate])

  bindMenuKeyHandler(useCallback((key: KeyEvent) => {
    if (key.defaultPrevented) return
    if (detail) {
      if (key.name === "escape") {
        key.preventDefault()
        setDetailId(null)
        setRootSwitchingEnabled(true)
      } else if (key.name === "up" && detailActions.length > 0) {
        key.preventDefault()
        setDetailActionIndex((current) => Math.max(0, Math.min(detailActions.length - 1, current - 1)))
      } else if (key.name === "down" && detailActions.length > 0) {
        key.preventDefault()
        setDetailActionIndex((current) => Math.min(detailActions.length - 1, current + 1))
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
        const candidate = candidates.find(({ id }) => id === pendingDeleteId)
        if (candidate?.download._tag === "Downloaded") local.deleteCatalogModel(candidate.id)
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
      setCursorId(candidates[Math.max(0, cursorIndex - 1)]!.id)
    } else if ((key.name === "down" || key.name === "j") && candidates.length > 0) {
      key.preventDefault()
      setCursorId(candidates[Math.min(candidates.length - 1, cursorIndex + 1)]!.id)
    } else if ((key.name === "return" || key.name === "enter") && cursor) {
      key.preventDefault()
      setDetailActionIndex(0)
      setDetailId(cursor.id)
      setRootSwitchingEnabled(false)
    } else if (key.name === "d" && cursor) {
      key.preventDefault()
      primaryAction(cursor)
    } else if (key.name === "s" && cursor && cursor.preparation._tag === "Installed") {
      key.preventDefault()
      selectCandidate(cursor)
    } else if (key.name === "backspace" && cursor) {
      if (cursor.download._tag === "Downloading") {
        local.cancelCatalogModelDownload(cursor.id)
        key.preventDefault()
      } else if (cursor.download._tag === "Downloaded") {
        setPendingDeleteId(cursor.id)
        key.preventDefault()
      }
    }
  }, [candidates, cursor, cursorIndex, detail, detailActions.length, focusedDetailAction, local, pendingDeleteId, primaryAction, runDetailAction, selectCandidate, setRootSwitchingEnabled]))

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
            {formatBytes(detail.runtimeMemoryBytes)} memory · {detail.quantization} · intelligence {Math.round(detail.intelligenceScore)}/100 · {qualityLabel(detail)}
          </text>
          <text style={{ fg: theme.muted }}>
            {Option.match(detail.estimatedTokensPerSecond, { onNone: () => "Speed unavailable", onSome: (speed) => `Approximately ${speed.toFixed(1)} tokens/sec` })}
          </text>
          <text style={{ fg: failed ? theme.error : downloading || downloaded ? theme.primary : theme.muted }}>
            {catalogStatus(detail)}
          </text>
          {failed && <text style={{ fg: theme.error }}>{detail.download.failure.message}</text>}
          <box style={{ paddingTop: 1, flexDirection: "column" }}>
            {detailActions.map((action, index) => (
              <MenuAction
                key={action}
                label={detailActionLabel[action]}
                focused={index === Math.min(detailActionIndex, Math.max(0, detailActions.length - 1))}
                tone={action === "primary" || action === "select" ? "primary" : "warning"}
                onClick={() => runDetailAction(action)}
                onMouseOver={() => setDetailActionIndex(index)}
              />
            ))}
          </box>
          <text style={{ fg: theme.muted, marginTop: 1 }}>License: {detail.license}</text>
          {detail.qualityEvidence.map((evidence) => <text key={evidence} style={{ fg: theme.muted }}>{evidence}</text>)}
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
        {runningProgress && <text style={{ fg: theme.primary }}>⠹ {runningProgress.label}{runningProgress.metadata}</text>}
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
        {candidates.length === 0 ? (
          <text style={{ fg: theme.warning }}>No compatible recommended models are currently available.</text>
        ) : candidates.map((candidate, index) => {
          const focused = index === cursorIndex
          const pendingDelete = pendingDeleteId === candidate.id
          return (
            <Button
              key={candidate.id}
              onClick={() => {
                setPendingDeleteId(null)
                setDetailActionIndex(0)
                setDetailId(candidate.id)
                setRootSwitchingEnabled(false)
              }}
              onMouseOver={() => {
                setCursorId(candidate.id)
                if (pendingDeleteId !== candidate.id) setPendingDeleteId(null)
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
              <text style={{ fg: theme.muted, width: 12 }}>{formatBytes(candidate.runtimeMemoryBytes)}</text>
              <text style={{ fg: theme.muted, width: 14 }}>{Math.round(candidate.intelligenceScore)}/100</text>
              <text style={{ fg: theme.muted, width: 14 }}>{qualityLabel(candidate)}</text>
              <text style={{ fg: theme.muted, width: 12 }}>{Option.match(candidate.estimatedTokensPerSecond, { onNone: () => "—", onSome: (speed) => `~${speed.toFixed(0)} t/s` })}</text>
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
        {Option.isSome(local.mutationFailure) && <text style={{ fg: theme.error }}>{Cause.pretty(local.mutationFailure.value.cause)}</text>}
      </scrollbox>
    </>
  )
})

const HardwareMenu = memo(function HardwareMenu({
  bindMenuKeyHandler,
}: MenuRootProps) {
  const theme = useTheme()
  const hardwareState = useLocalInferenceHardware()
  const snapshot = Result.value(hardwareState)

  bindMenuKeyHandler(useCallback(() => {}, []))

  return (
    <>
      <MenuHeader title="Hardware" subtitle="View detected hardware" />
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
        {Option.match(snapshot, {
          onNone: () => (
            <text style={{ fg: Result.isFailure(hardwareState) ? theme.error : theme.muted }}>
              {Result.isFailure(hardwareState) ? "Hardware detection is unavailable." : "Detecting local-inference hardware…"}
            </text>
          ),
          onSome: ({ state: detectedHardware }) => {
            const hardware = describeLocalHardware(detectedHardware)
            const memory = deriveHardwareMemoryView(detectedHardware)
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
                <box style={{ flexDirection: "column", paddingTop: 1 }}>
                  {memory.domains.map((domain) => <HardwareMemoryDomain key={domain.id} domain={domain} />)}
                </box>
              </>
            )
          },
        })}
      </scrollbox>
    </>
  )
})

const CloudMenu = memo(function CloudMenu({
  setRootSwitchingEnabled,
  bindMenuKeyHandler,
}: MenuRootProps) {
  const theme = useTheme()
  const platform = usePlatform()
  const settings = useSettingsState()
  const config = useModelConfig()
  const authSource = useAtomValue(authSourceAtom)
  const [mode, setMode] = useState<"root" | "edit" | "disconnect">("root")
  const [keyValue, setKeyValue] = useState("")
  const [validationError, setValidationError] = useState<string | null>(null)
  const [cursorIndex, setCursorIndex] = useState(0)
  const [disconnectActionIndex, setDisconnectActionIndex] = useState(0)
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
  const selectedAction = actionIds[Math.min(cursorIndex, actionIds.length - 1)]

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
      setDisconnectActionIndex(0)
      setMode("disconnect")
      setRootSwitchingEnabled(false)
      return
    }
    void platform.openLink(MAGNITUDE_CLOUD_URL)
  }, [platform, setRootSwitchingEnabled])

  bindMenuKeyHandler(useCallback((key: KeyEvent) => {
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
        setDisconnectActionIndex((current) => Math.max(0, current - 1))
        return
      }
      if (key.name === "down") {
        key.preventDefault()
        setDisconnectActionIndex((current) => Math.min(1, current + 1))
        return
      }
      if (key.name === "return" || key.name === "enter") {
        key.preventDefault()
        if (disconnectActionIndex === 1) auth.clear()
        setMode("root")
        setRootSwitchingEnabled(true)
      }
      return
    }
    if (key.name === "up" && actionIds.length > 0) {
      key.preventDefault()
      setCursorIndex((current) => Math.max(0, Math.min(actionIds.length - 1, current - 1)))
      return
    }
    if (key.name === "down" && actionIds.length > 0) {
      key.preventDefault()
      setCursorIndex((current) => Math.min(actionIds.length - 1, current + 1))
      return
    }
    if ((key.name === "return" || key.name === "enter") && selectedAction) {
      key.preventDefault()
      runAction(selectedAction)
    }
  }, [actionIds.length, auth, disconnectActionIndex, mode, runAction, save, selectedAction, setRootSwitchingEnabled]))

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
              focused={disconnectActionIndex === 0}
              onClick={() => {
                setMode("root")
                setRootSwitchingEnabled(true)
              }}
              onMouseOver={() => setDisconnectActionIndex(0)}
            />
            <MenuAction
              label="Disconnect"
              focused={disconnectActionIndex === 1}
              tone="error"
              onClick={() => {
                auth.clear()
                setMode("root")
                setRootSwitchingEnabled(true)
              }}
              onMouseOver={() => setDisconnectActionIndex(1)}
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
              onMouseOver={() => setCursorIndex(actionIds.indexOf("add"))}
            >
              <text style={{ fg: theme.primary }}>{selectedAction === "add" ? "› " : "  "}Add API key</text>
            </Button>
          )}
          {auth.source === "config" && (
            <>
              <Button
                onClick={() => runAction("update")}
                onMouseOver={() => setCursorIndex(actionIds.indexOf("update"))}
              >
                <text style={{ fg: selectedAction === "update" ? theme.primary : theme.foreground }}>
                  {selectedAction === "update" ? "› " : "  "}Update API key
                </text>
              </Button>
              <Button
                onClick={() => runAction("disconnect")}
                onMouseOver={() => setCursorIndex(actionIds.indexOf("disconnect"))}
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
              onMouseOver={() => setCursorIndex(actionIds.indexOf("link"))}
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
