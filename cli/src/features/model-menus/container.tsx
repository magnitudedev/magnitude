import {
  memo,
  useCallback,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react"
import { TextAttributes, type KeyEvent, type ScrollBoxRenderable } from "@opentui/core"
import { useKeyboard } from "@opentui/react"
import { Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Cause, Option } from "effect"
import {
  deriveHardwareMemoryView,
  deriveCurrentLocalModel,
  formatLocalModelDisplayName,
  formatModelDisplayName,
  modelSlotResidentAllocation,
  getDisplayWidth,
  getAnimationTimeSnapshot,
  localModelConfigurationId,
  localModelProviderModelId,
  localModelCapabilities,
  localModelSpeculativeMethodLabel,
  truncateToDisplayWidth,
  type NotificationState,
  usePlatform,
  useLocalInferenceHardware,
  useCatalogModels,
  useLocalModelActions,
  useLocalModelsSelector,
  useModelSlotActions,
  usePreviewModelLoad,
  useModelConfig,
  useSettingsState,
  type CatalogModelReconciliationState,
} from "@magnitudedev/client-common"
import {
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelCatalogLifecycle,
  ReasoningEffortSchema,
  servableModelBundlePackages,
  type LocalModel,
  type ModelSlotsState,
  type ProviderCatalogEntry,
  type ProviderModelDisabledReason,
  type ProviderModelId,
  type ModelServingConfigurationId,
  type ProviderModelCatalogEntry,
  type ReasoningEffort,
} from "@magnitudedev/sdk"
import { Button } from "../../components/button"
import { HardwareMemoryDomain } from "../../components/hardware-memory-domain"
import { useSpinnerFrame } from "../../hooks/use-spinner-frame"
import { useBoundedCursor } from "../../hooks/use-bounded-cursor"
import { useLocalWidth } from "../../hooks/use-local-width"
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
  modelDownloadFailureMessage,
  localModelMaximumContextLength,
  localModelBundleKey,
  localInferenceProgressLines,
  performanceRangeSpeedLabel,
} from "../local-inference/view-model"
import { deriveSettingsAuthInfo } from "../overlays/auth-display"
import {
  catalogDetailHints,
  catalogListHints,
  CATALOG_INSPECTOR_CONTENT_WIDTH,
  CATALOG_SPLIT_INSPECTOR_HEIGHTS,
  deriveCatalogLayout,
  formatCatalogModelLabel,
  type CatalogLayout,
} from "./catalog-layout"
import {
  pentagonRadarValues,
  retargetPentagonRadar,
  type PentagonRadarTransition,
} from "../../components/pentagon-radar"
import { localModelRadarAxes } from "../local-inference/model-radar"
import { formatModelClassification } from "../local-inference/model-classification"
import { CatalogRadarView } from "./catalog-radar-view"
import {
  modelMenusLocalModelsStateEquivalent,
  selectModelMenusLocalModelsState,
} from "./state"
import { NotificationArea } from "../notification-area/notification-area"

// Cloud is disabled.
const ROOTS = ["models", "catalog", "hardware"] as const
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
  // { label: "Connect cloud models", root: "cloud" },
] as const satisfies readonly { readonly label: string; readonly root: ModelMenuRoot }[]

interface ModelsMenuProps {
  readonly openRoot: (root: ModelMenuRoot) => void
  readonly openCatalogDetail: (providerModelId: string) => void
  readonly setRootSwitchingEnabled: (enabled: boolean) => void
}

interface ModelsMenuOrdering {
  readonly selectedModel: Option.Option<Pick<ProviderModelCatalogEntry, "providerId" | "providerModelId">>
  readonly recentModelKeys: readonly string[]
  readonly favoriteKeys: ReadonlySet<string>
}

interface CatalogMenuProps {
  readonly initialCatalogDetailId: string | null
  readonly setRootSwitchingEnabled: (enabled: boolean) => void
}

interface CloudMenuProps {
  readonly setRootSwitchingEnabled: (enabled: boolean) => void
}

const nextRoot = (root: ModelMenuRoot, direction: -1 | 1): ModelMenuRoot => {
  const index = ROOTS.findIndex((candidate) => candidate === root)
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

const catalogCandidateRowId = (configurationId: string): string =>
  `catalog-candidate:${configurationId}`

export const scrollCatalogCandidateIntoView = (
  scrollbox: Pick<ScrollBoxRenderable, "scrollChildIntoView"> | null,
  configurationId: string,
): void => {
  scrollbox?.scrollChildIntoView(catalogCandidateRowId(configurationId))
}

const formatContextWindow = (tokens: number): string =>
  tokens >= 1_000_000
    ? `${(tokens / 1_000_000).toFixed(1)}M`
    : tokens >= 1_000
      ? `${Math.round(tokens / 1_000)}K`
      : String(tokens)

const providerModelKey = (model: Pick<ProviderModelCatalogEntry, "providerId" | "providerModelId">): string =>
  `${model.providerId}:${model.providerModelId}`

interface CatalogContents {
  readonly providers: readonly ProviderCatalogEntry[]
  readonly models: readonly ProviderModelCatalogEntry[]
}

const catalogContents = (
  config: ReturnType<typeof useModelConfig>,
): CatalogContents => Option.getOrElse(
  Option.map(Result.value(config.catalog), ({ state }) =>
    ProviderModelCatalogLifecycle.match(state, {
      Loading: () => ({ providers: [], models: [] }),
      Ready: ({ providers, models }) => ({ providers, models }),
      Refreshing: ({ providers, models }) => ({ providers, models }),
      Degraded: ({ providers, models }) => ({ providers, models }),
      Unavailable: ({ providers }) => ({ providers, models: [] }),
    })),
  () => ({ providers: [], models: [] }),
)

const catalogModels = (config: ReturnType<typeof useModelConfig>) =>
  catalogContents(config).models

const providerKindLabel = (kind: ProviderCatalogEntry["kind"]): string => {
  switch (kind) {
    case "Custom": return "Custom"
    case "Local": return "Local"
    case "Hosted": return "Cloud"
  }
}

export const providerDisabledStatus = (reason: ProviderModelDisabledReason): string => {
  switch (reason) {
    case "insufficient_resources": return "Insufficient resources"
    case "provider_unavailable": return "Provider unavailable"
    case "model_unavailable": return "Model unavailable"
    case "installation_unavailable": return "Installation missing"
    case "incompatible_runtime": return "Incompatible runtime"
    case "invalid_configuration": return "Invalid configuration"
    default: {
      const unhandled: never = reason
      return unhandled
    }
  }
}

const installedOriginStatus = (
  origins: readonly ("Magnitude" | "HuggingFaceCache")[],
): string => origins.every((origin) => origin === "HuggingFaceCache")
  ? "Installed (HF)"
  : "Installed"

export const localModelInstalledStatus = (
  model: LocalModel,
): string => model.acquisitionState._tag === "Installed"
  ? installedOriginStatus(model.acquisitionState.packages.map(({ origin }) => origin))
  : "Installed"

export const localModelReadinessStatus = (
  model: LocalModel,
): string => {
  if (model.servingState._tag === "Resolving") return "Resolving"
  if (model.servingState._tag === "Assessing") return "Assessing"
  if (model.servingState._tag === "Failed") return "Error"

  const assessment = model.servingState.assessment
  if (assessment._tag === "Incompatible") return "Error"
  if (assessment._tag === "DoesNotFit") return "Doesn’t fit"
  if (model.upgradeState._tag === "Available") return "Update available"
  if (model.upgradeState._tag === "Upgrading") return "Updating"
  if (model.upgradeState._tag === "Failed") return "Update error"
  return model.acquisitionState._tag === "Installed"
    ? localModelInstalledStatus(model)
    : "Available"
}

export type ModelsMenuEntry =
  | {
      readonly _tag: "Local"
      readonly id: string
      readonly model: LocalModel
    }
  | {
      readonly _tag: "LocalStatus"
      readonly id: string
      readonly model: LocalModel
    }
  | {
      readonly _tag: "Provider"
      readonly id: string
      readonly model: ProviderModelCatalogEntry
      readonly provider: ProviderCatalogEntry
    }

export const modelsMenuStatusPresentation = (
  status: string,
): { readonly label: string; readonly tone: "muted" | "warning" } =>
  status === "Low free memory"
    ? { label: `! ${status}`, tone: "warning" }
    : { label: status, tone: "muted" }

const modelsMenuProviderModel = (
  entry: ModelsMenuEntry,
): ProviderModelCatalogEntry | undefined => entry._tag === "Provider"
  ? entry.model
  : undefined

const modelsMenuLocalModel = (entry: ModelsMenuEntry): LocalModel | undefined =>
  entry._tag === "Local" ? entry.model
    : entry._tag === "LocalStatus" ? entry.model : undefined

const modelsMenuDisplayName = (entry: ModelsMenuEntry): string => entry._tag === "Local"
  ? formatLocalModelDisplayName(entry.model)
  : entry._tag === "LocalStatus"
    ? formatLocalModelDisplayName(entry.model)
    : formatModelDisplayName(entry.model.displayName, entry.model.variantLabel)

const modelsMenuContextLength = (entry: ModelsMenuEntry): Option.Option<number> => entry._tag === "Local"
  ? entry.model.servingState._tag === "Assessed"
    ? Option.some(entry.model.servingState.configuration.profile.contextLength)
    : localModelMaximumContextLength(entry.model)
  : entry._tag === "Provider"
    ? Option.some(entry.model.contextWindow)
    : localModelMaximumContextLength(entry.model)

const modelsMenuContextLabel = (entry: ModelsMenuEntry): string => Option.match(
  modelsMenuContextLength(entry),
  {
    onNone: () => "Unknown",
    onSome: formatContextWindow,
  },
)

const modelsMenuOfferingKey = (entry: ModelsMenuEntry): Option.Option<string> => {
  const providerModel = modelsMenuProviderModel(entry)
  if (providerModel !== undefined) return Option.some(providerModelKey(providerModel))
  const localModel = modelsMenuLocalModel(entry)
  return localModel === undefined
    ? Option.none()
    : Option.map(localModelProviderModelId(localModel), (providerModelId) =>
        `${LOCAL_PROVIDER_ID}:${providerModelId}`)
}

const modelsMenuProviderIdentity = (
  entry: ModelsMenuEntry,
): Option.Option<Pick<ProviderModelCatalogEntry, "providerId" | "providerModelId">> => {
  const providerModel = modelsMenuProviderModel(entry)
  if (providerModel !== undefined) return Option.some(providerModel)
  const localModel = modelsMenuLocalModel(entry)
  return localModel === undefined
    ? Option.none()
    : Option.map(localModelProviderModelId(localModel), (providerModelId) => ({
        providerId: LOCAL_PROVIDER_ID,
        providerModelId,
      }))
}

const sameProviderModel = (
  left: Pick<ProviderModelCatalogEntry, "providerId" | "providerModelId">,
  right: Pick<ProviderModelCatalogEntry, "providerId" | "providerModelId">,
): boolean => left.providerId === right.providerId
  && left.providerModelId === right.providerModelId

export const modelsMenuEntryIsSelected = (
  entry: ModelsMenuEntry,
  selectedModel: Option.Option<Pick<ProviderModelCatalogEntry, "providerId" | "providerModelId">>,
): boolean => Option.exists(
  selectedModel,
  (selected) => entry._tag === "Provider"
    ? sameProviderModel(entry.model, selected)
    : Option.exists(
        Option.fromNullable(modelsMenuLocalModel(entry)),
        (model) => Option.contains(localModelProviderModelId(model), selected.providerModelId)
          && selected.providerId === LOCAL_PROVIDER_ID,
      ),
)

export const modelsMenuEntryIsEligible = (
  entry: ModelsMenuEntry,
  selectedAtOpen: Option.Option<Pick<ProviderModelCatalogEntry, "providerId" | "providerModelId">>,
): boolean => {
  const providerModel = modelsMenuProviderModel(entry)
  if (entry._tag !== "Provider") return true
  return providerModel !== undefined
    && providerModel.supportedSlots.includes(PRIMARY_SLOT_ID)
    && (providerModel.availability._tag === "Available"
      || modelsMenuEntryIsSelected(entry, selectedAtOpen))
}

export const modelsMenuOrderingAtOpen = (
  catalogReady: boolean,
  slotsReady: boolean,
  slots: Option.Option<ModelSlotsState>,
  selected: Option.Option<Pick<ProviderModelCatalogEntry, "providerId" | "providerModelId">>,
): Option.Option<ModelsMenuOrdering> => !catalogReady || !slotsReady
  ? Option.none()
  : Option.some({
      selectedModel: selected,
      recentModelKeys: Option.match(slots, {
        onNone: () => [],
        onSome: ({ recentModels }) => recentModels.primary.map(providerModelKey),
      }),
      favoriteKeys: new Set(Option.match(slots, {
        onNone: () => [],
        onSome: ({ favoriteModels }) => favoriteModels.map(providerModelKey),
      })),
    })

export const buildModelsMenuEntries = (
  localModels: readonly LocalModel[],
  providerModels: readonly ProviderModelCatalogEntry[],
  providers: readonly ProviderCatalogEntry[],
): readonly ModelsMenuEntry[] => {
  return [
    ...localModels.flatMap((model): readonly ModelsMenuEntry[] => {
      if (model.acquisitionState._tag !== "Installed") return []
      const bundleKey = localModelBundleKey(model)
      return [{
        _tag: model.servingState._tag === "Assessed"
          && model.servingState.assessment._tag === "Fits"
          ? "Local" : "LocalStatus",
        id: `local:${bundleKey}:status`,
        model,
      }]
    }),
    ...providerModels.flatMap((model): readonly ModelsMenuEntry[] => {
      if (model.providerId === LOCAL_PROVIDER_ID) return []
      const provider = providers.find(({ providerId }) => providerId === model.providerId)
      return provider === undefined ? [] : [{
        _tag: "Provider",
        id: providerModelKey(model),
        model,
        provider,
      }]
    }),
  ]
}

export const catalogLocalModels = (
  models: readonly LocalModel[],
): readonly LocalModel[] => models.filter((model) =>
  model.catalogMembershipState._tag === "InCatalog"
  && model.servingState._tag === "Assessed"
  && model.servingState.assessment._tag === "Fits")

export type ModelsMenuSelectionAction =
  | {
      readonly _tag: "AssignProvider"
      readonly providerModel: ProviderModelCatalogEntry
    }
  | {
      readonly _tag: "AssignLocal"
      readonly providerModelId: ProviderModelId
      readonly reasoningEffort: Option.Option<ReasoningEffort>
    }
  | {
      readonly _tag: "InstallConfiguration"
      readonly configurationId: ModelServingConfigurationId
      readonly reasoningEffort: Option.Option<ReasoningEffort>
    }

export const modelsMenuSelectionAction = (
  entry: ModelsMenuEntry,
): Option.Option<ModelsMenuSelectionAction> => {
  const providerModel = modelsMenuProviderModel(entry)
  if (providerModel !== undefined) {
    return providerModel.supportedSlots.includes(PRIMARY_SLOT_ID)
      && providerModel.availability._tag === "Available"
      ? Option.some({ _tag: "AssignProvider", providerModel })
      : Option.none()
  }
  if (entry._tag === "Provider"
    || entry._tag === "LocalStatus") return Option.none()
  const { model } = entry
  if (model.servingState._tag !== "Assessed"
    || model.servingState.assessment._tag !== "Fits") return Option.none()
  const reasoningEffort = model.servingState.capabilities.reasoning.defaultEffort
  if (model.servingState.availabilityState._tag === "Selectable") {
    return Option.some({
      _tag: "AssignLocal",
      providerModelId: model.servingState.availabilityState.providerModelId,
      reasoningEffort,
    })
  }
  if (model.servingState.availabilityState._tag !== "Installable") return Option.none()
  return Option.some({
    _tag: "InstallConfiguration",
    configurationId: model.servingState.configuration.id,
    reasoningEffort,
  })
}

export const ModelMenusContainer = memo(function ModelMenusContainer({
  notificationState,
}: {
  readonly notificationState: NotificationState | null
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
        backgroundColor: theme.background.canvas,
      }}
    >
      <box style={{ flexGrow: 1, minHeight: 0, flexDirection: "column", backgroundColor: theme.background.menu }}>
        {menu.root === "models" && <ModelsMenu openRoot={openRoot} openCatalogDetail={openCatalogDetail} setRootSwitchingEnabled={setAtRootLevel} />}
        {menu.root === "catalog" && <CatalogMenu initialCatalogDetailId={catalogDetailId} setRootSwitchingEnabled={setAtRootLevel} />}
        {menu.root === "hardware" && <HardwareMenu />}
        {/* Cloud is disabled. */}
        {/* {menu.root === "cloud" && <CloudMenu setRootSwitchingEnabled={setAtRootLevel} />} */}
      </box>
      <box
        style={{
          height: 1,
          flexShrink: 0,
          borderStyle: "single",
          border: ["bottom"],
          borderColor: theme.background.menu,
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
          backgroundColor: theme.background.canvas,
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
                  fg: active ? theme.background.menu : theme.text.body,
                  ...(active ? { bg: theme.text.body } : {}),
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
        {notificationState !== null && (
          <NotificationArea
            notificationState={notificationState}
            theme={theme}
            onAction={(action) => {
              if (action === "openCatalog") openRoot("catalog")
            }}
          />
        )}
        <box style={{ flexGrow: 1 }} />
        <text style={{ fg: theme.text.metadata }}>
          {atRootLevel ? "←/→ switch menus" : "←/→ switch menus · Esc back"}
        </text>
      </box>
    </box>
  )
})

const MenuHeader = memo(function MenuHeader({
  title,
  subtitle,
  selection,
  onSectionClick,
  summary,
  hints,
  compact = false,
  width,
}: {
  readonly title: string
  readonly subtitle?: string
  readonly selection?: string
  readonly onSectionClick?: () => void
  readonly summary?: string
  readonly hints?: string
  readonly compact?: boolean
  readonly width?: number
}) {
  const theme = useTheme()
  const [sectionHovered, setSectionHovered] = useState(false)
  const sectionTitle = (
    <text
      style={{ fg: theme.text.body }}
      attributes={TextAttributes.BOLD | (sectionHovered ? TextAttributes.UNDERLINE : TextAttributes.NONE)}
    >
      {title.toUpperCase()}
    </text>
  )
  const compactSelectionWidth = Math.max(1, (width ?? 80) - 4)
  const displayedHints = width === undefined || hints === undefined
    ? hints
    : truncateToDisplayWidth(hints, compactSelectionWidth)
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
        {!compact && subtitle && <text style={{ fg: theme.text.metadata }}> · {subtitle}</text>}
        {!compact && selection && <text style={{ fg: theme.text.body }}> → {selection}</text>}
        <box style={{ flexGrow: 1 }} />
        {summary && <text style={{ fg: theme.text.metadata }}>{summary}</text>}
      </box>
      {compact && selection && (
        <text style={{ fg: theme.text.body }} wrapMode="none">
          {truncateToDisplayWidth(selection, compactSelectionWidth)}
        </text>
      )}
      {displayedHints && <text style={{ fg: theme.text.metadata }} wrapMode="none">{displayedHints}</text>}
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
  onMouseOut,
}: {
  readonly label: string
  readonly focused: boolean
  readonly tone?: MenuActionTone
  readonly onClick: () => void
  readonly onMouseOver: () => void
  readonly onMouseOut?: () => void
}) {
  const theme = useTheme()
  const color = focused
    ? theme.accent
    : tone === "primary"
      ? theme.accent
      : tone === "link"
        ? theme.link
        : tone === "warning"
          ? theme.status.warning
          : tone === "error"
            ? theme.status.failure
            : theme.text.body
  return (
    <Button onClick={onClick} onMouseOver={onMouseOver} onMouseOut={onMouseOut}>
      <text style={{ fg: color }}>{focused ? "› " : "  "}{label}</text>
    </Button>
  )
})

const ModelsMenu = memo(function ModelsMenu(props: ModelsMenuProps) {
  const theme = useTheme()
  const config = useModelConfig()
  const catalogReady = Result.match(config.catalog, {
    onInitial: () => false,
    onFailure: () => true,
    onSuccess: ({ value: { state } }) => ProviderModelCatalogLifecycle.match(state, {
      Loading: () => false,
      Ready: () => true,
      Refreshing: () => true,
      Degraded: () => true,
      Unavailable: () => true,
    }),
  })
  const slotsReady = !Result.isInitial(config.slots)
  const slots = Option.map(Result.value(config.slots), ({ state }) => state)
  const selected = Option.flatMap(config.selections, ({ primary }) => primary)
  const ordering = modelsMenuOrderingAtOpen(catalogReady, slotsReady, slots, selected)

  return Option.match(ordering, {
    onNone: () => <text style={{ fg: theme.status.progress }}>Loading models…</text>,
    onSome: (initialOrdering) => (
      <ReadyModelsMenu {...props} config={config} initialOrdering={initialOrdering} />
    ),
  })
})

const ReadyModelsMenu = memo(function ReadyModelsMenu({
  openRoot,
  openCatalogDetail,
  setRootSwitchingEnabled,
  config,
  initialOrdering,
}: ModelsMenuProps & {
  readonly config: ReturnType<typeof useModelConfig>
  readonly initialOrdering: ModelsMenuOrdering
}) {
  const theme = useTheme()
  const localSnapshot = useLocalModelsSelector(
    selectModelMenusLocalModelsState,
    modelMenusLocalModelsStateEquivalent,
  )
  const modelActions = useLocalModelActions()
  const slotActions = useModelSlotActions()
  const catalog = catalogContents(config)
  const slotsSnapshot = Result.value(config.slots)
  const selectedModel = Option.flatMap(config.selections, ({ primary }) => primary)
  const currentFavoriteKeys = new Set(config.favoriteModels.map(providerModelKey))
  const [ordering] = useState(initialOrdering)
  const projectedLocalModels = Option.match(localSnapshot, {
    onNone: () => [] as readonly LocalModel[],
    onSome: ({ models }) => models,
  })
  const entries = buildModelsMenuEntries(
    projectedLocalModels,
    catalog.models,
    catalog.providers,
  )
  const isSelected = (entry: ModelsMenuEntry): boolean =>
    modelsMenuEntryIsSelected(entry, selectedModel)
  const isFavorite = (entry: ModelsMenuEntry): boolean =>
    Option.exists(modelsMenuOfferingKey(entry), (key) => currentFavoriteKeys.has(key))
  const isEligible = (entry: ModelsMenuEntry): boolean => {
    return modelsMenuEntryIsEligible(entry, ordering.selectedModel)
  }
  const eligible = entries
    .filter(isEligible)
    .sort((left, right) => {
      const leftKey = modelsMenuOfferingKey(left)
      const rightKey = modelsMenuOfferingKey(right)
      const leftFavorite = Option.exists(leftKey, (key) => ordering.favoriteKeys.has(key))
      const rightFavorite = Option.exists(rightKey, (key) => ordering.favoriteKeys.has(key))
      if (leftFavorite !== rightFavorite) return leftFavorite ? -1 : 1
      const leftSelected = modelsMenuEntryIsSelected(left, ordering.selectedModel)
      const rightSelected = modelsMenuEntryIsSelected(right, ordering.selectedModel)
      if (leftSelected !== rightSelected) return leftSelected ? -1 : 1
      const leftProviderModel = modelsMenuProviderModel(left)
      const rightProviderModel = modelsMenuProviderModel(right)
      const leftRecency = leftProviderModel === undefined
        ? -1
        : ordering.recentModelKeys.indexOf(providerModelKey(leftProviderModel))
      const rightRecency = rightProviderModel === undefined
        ? -1
        : ordering.recentModelKeys.indexOf(providerModelKey(rightProviderModel))
      if (leftRecency !== rightRecency) {
        if (leftRecency < 0) return 1
        if (rightRecency < 0) return -1
        return leftRecency - rightRecency
      }
      const leftLocal = left._tag !== "Provider"
      const rightLocal = right._tag !== "Provider"
      if (leftLocal !== rightLocal) return leftLocal ? -1 : 1
      return modelsMenuDisplayName(left).localeCompare(modelsMenuDisplayName(right))
        || left.id.localeCompare(right.id)
    })
  const [cursorId, setCursorId] = useState<string | null>(null)
  const [detailId, setDetailId] = useState<string | null>(null)
  const cursorIndex = Math.max(0, eligible.findIndex(({ id }) => id === cursorId))
  const cursor = eligible[cursorIndex]
  const detail = eligible.find(({ id }) => id === detailId) ?? null
  const memoryFor = (entry: ModelsMenuEntry) => {
    const model = modelsMenuLocalModel(entry)
    return model?.servingState._tag === "Assessed"
      && model.servingState.assessment._tag === "Fits"
      ? Option.some(model.servingState.assessment.memory)
      : Option.none()
  }
  const requirementFor = (entry: ModelsMenuEntry): string => {
    if (entry._tag === "Provider") {
      return providerKindLabel(entry.provider.kind)
    }
    if (entry._tag === "LocalStatus") {
      const assessment = entry.model.servingState._tag === "Assessed"
        ? entry.model.servingState.assessment
        : undefined
      if (assessment?._tag === "DoesNotFit") {
        return formatBytes(assessment.totalRequiredBytes)
      }
      return Option.match(memoryFor(entry), {
        onNone: () => "—",
        onSome: ({ totalRequiredBytes }) => formatBytes(totalRequiredBytes),
      })
    }
    const assessment = entry.model.servingState._tag === "Assessed"
      ? entry.model.servingState.assessment
      : undefined
    if (assessment?._tag === "DoesNotFit") {
      return formatBytes(assessment.totalRequiredBytes)
    }
    return Option.match(memoryFor(entry), {
      onNone: () => "—",
      onSome: ({ totalRequiredBytes }) => formatBytes(totalRequiredBytes),
    })
  }
  const primarySlot = Option.match(slotsSnapshot, {
    onNone: () => null,
    onSome: ({ state }) => state.slots.primary,
  })
  const detailIsLocal = detail !== null && detail._tag !== "Provider"
  const detailIsSelected = detail !== null && isSelected(detail)
  const detailLocalModel = detail === null ? undefined : modelsMenuLocalModel(detail)
  const detailCatalogConfigurationId = detailLocalModel?.servingState._tag === "Assessed"
    && detailLocalModel.catalogMembershipState._tag === "InCatalog"
    ? detailLocalModel.servingState.configuration.id
    : undefined
  const detailActions = useMemo(() => {
    if (!detail) return [] as readonly ("select" | "load" | "stop" | "catalog")[]
    const actions: ("select" | "load" | "stop" | "catalog")[] = []
    if (!detailIsSelected && Option.isSome(modelsMenuSelectionAction(detail))) actions.push("select")
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
    if (detailCatalogConfigurationId) actions.push("catalog")
    return actions
  }, [detail, detailCatalogConfigurationId, detailIsLocal, detailIsSelected, primarySlot])
  const detailActionCursor = useBoundedCursor(detailActions.length)
  const emptyActionCursor = useBoundedCursor(EMPTY_MODEL_ACTIONS.length)
  const focusedDetailAction = detailActions[detailActionCursor.index]
  const installationFailed = modelActions.latestInstallationFailed

  const statusFor = (entry: ModelsMenuEntry): string => {
    const selectedEntry = isSelected(entry)
    if (entry._tag === "LocalStatus") {
      return localModelReadinessStatus(entry.model)
    }
    if (entry._tag === "Local") {
      const memory = memoryFor(entry)
      if (selectedEntry
        && primarySlot?._tag === "ConfiguredLocal"
        && (primarySlot.residency._tag === "Loading"
          || primarySlot.residency._tag === "Ready"
          || primarySlot.residency._tag === "Stopping")
        && Option.exists(memory, ({ currentHeadroomState }) =>
          currentHeadroomState._tag === "Insufficient")) return "Selected"
      if (Option.exists(memory, ({ currentHeadroomState }) =>
        currentHeadroomState._tag === "Insufficient")) return "Low free memory"
      if (Option.exists(memory, ({ systemUseState }) => systemUseState._tag === "High")) return "Tight fit"
      const servingState = entry.model.servingState
      if (servingState._tag === "Assessed"
        && servingState.availabilityState._tag === "Unavailable") {
        return servingState.availabilityState.failure.message
      }
    }
    if (selectedEntry) return "Selected"
    return entry._tag === "Local"
      ? localModelInstalledStatus(entry.model)
      : "Available"
  }

  const choose = useCallback((entry: ModelsMenuEntry) => {
    const action = modelsMenuSelectionAction(entry)
    if (Option.isNone(action)) return
    if (action.value._tag === "AssignProvider") {
      const { providerModel } = action.value
      config.updateSlotModel(PRIMARY_SLOT_ID, providerModel.providerId, providerModel.providerModelId)
      return
    }
    if (action.value._tag === "AssignLocal") {
      slotActions.assign(PRIMARY_SLOT_ID, {
        providerId: LOCAL_PROVIDER_ID,
        providerModelId: action.value.providerModelId,
        reasoningEffort: Option.getOrElse(
          action.value.reasoningEffort,
          () => ReasoningEffortSchema.make("none"),
        ),
      })
      return
    }
    const createAction = action.value
    modelActions.installAndAssign(
      createAction.configurationId,
      PRIMARY_SLOT_ID,
      Option.getOrElse(
        createAction.reasoningEffort,
        () => ReasoningEffortSchema.make("none"),
      ),
    )
  }, [config, modelActions, slotActions])

  const toggleFavorite = useCallback((entry: ModelsMenuEntry) => {
    Option.match(modelsMenuProviderIdentity(entry), {
      onNone: () => {},
      onSome: (model) => config.setModelFavorite(
        model,
        !currentFavoriteKeys.has(providerModelKey(model)),
      ),
    })
  }, [config, currentFavoriteKeys])

  const runDetailAction = useCallback((action: typeof detailActions[number]) => {
    if (!detail) return
    if (action === "select") choose(detail)
    else if (action === "load" && primarySlot?._tag === "ConfiguredLocal") {
      void slotActions.load(PRIMARY_SLOT_ID)
    }
    else if (action === "stop" && primarySlot?._tag === "ConfiguredLocal") {
      void slotActions.stop(PRIMARY_SLOT_ID)
    }
    else if (detailCatalogConfigurationId) openCatalogDetail(detailCatalogConfigurationId)
  }, [choose, detail, detailCatalogConfigurationId, openCatalogDetail, primarySlot, slotActions])

  useKeyboard(useCallback((key: KeyEvent) => {
    if (key.defaultPrevented) return
    if (detail) {
      if (key.name === "f" && !key.ctrl && !key.meta && !key.option
        && Option.isSome(modelsMenuProviderIdentity(detail))) {
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
      setCursorId(eligible[Math.max(0, cursorIndex - 1)]!.id)
      return
    }
    if ((key.name === "down" || key.name === "j") && eligible.length > 0) {
      key.preventDefault()
      setCursorId(eligible[Math.min(eligible.length - 1, cursorIndex + 1)]!.id)
      return
    }
    if ((key.name === "return" || key.name === "enter") && cursor) {
      key.preventDefault()
      choose(cursor)
      return
    }
    if (key.name === "f" && !key.ctrl && !key.meta && !key.option && cursor
      && Option.isSome(modelsMenuProviderIdentity(cursor))) {
      key.preventDefault()
      toggleFavorite(cursor)
      return
    }
    if (key.name === "d" && cursor) {
      key.preventDefault()
      detailActionCursor.reset()
      setDetailId(cursor.id)
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
    const detailProviderModel = modelsMenuProviderModel(detail)
    const detailHasProviderIdentity = Option.isSome(modelsMenuProviderIdentity(detail))
    const detailCapabilities = detailProviderModel?.capabilities
      ?? (detail._tag === "LocalStatus"
        ? Option.getOrUndefined(localModelCapabilities(detail.model))
        : detail._tag === "Local"
          ? Option.getOrUndefined(localModelCapabilities(detail.model))
          : undefined)
    const detailFavorite = isFavorite(detail)
    const detailMemory = memoryFor(detail)
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
          selection={modelsMenuDisplayName(detail)}
          onSectionClick={() => {
            setDetailId(null)
            setRootSwitchingEnabled(true)
          }}
          hints={detailActions.length > 0
            ? !detailHasProviderIdentity
              ? "↑↓ navigate · Enter choose · Esc back"
              : "↑↓ navigate · Enter choose · F favorite · Esc back"
            : !detailHasProviderIdentity ? "Esc back" : "F favorite · Esc back"}
        />
        <box style={{ flexGrow: 1, minHeight: 0, flexDirection: "column", paddingLeft: 2, paddingRight: 2, paddingTop: 1 }}>
          <text style={{ fg: theme.text.body }} attributes={TextAttributes.BOLD}>
            {detailFavorite ? "★ " : ""}{modelsMenuDisplayName(detail)}
          </text>
          <text style={{ fg: theme.text.metadata }}>
            {detail._tag === "Provider" ? providerKindLabel(detail.provider.kind) : "Local"} · {modelsMenuContextLabel(detail)} context · {statusFor(detail)}
          </text>
          {detailLocalModel && (
            <>
              {detailLocalModel.catalogMembershipState._tag === "InCatalog" && detailLocalModel.catalogMembershipState.catalogData.quantizationAware && (
                <text style={{ fg: theme.text.metadata }}>Training: Quantization-aware</text>
              )}
            </>
          )}
          {detailIsLocal && Option.exists(detailMemory, ({ currentHeadroomState, systemUseState }) =>
            currentHeadroomState._tag === "Insufficient" || systemUseState._tag === "High") && (
            <text style={{ fg: theme.status.warning }}>
              {Option.exists(detailMemory, ({ currentHeadroomState }) =>
                currentHeadroomState._tag === "Insufficient")
                ? "Not enough free memory right now"
                : "Uses more than the recommended share of system memory"}
            </text>
          )}
          {detailCapabilities && (
            <text style={{ fg: theme.text.metadata }}>
              {detailCapabilities.vision ? "Vision" : "No vision"} · Tools · {detailCapabilities.reasoning.supported ? "Reasoning" : "No reasoning"}
            </text>
          )}
          <box style={{ paddingTop: 1, flexDirection: "column" }}>
            {detailIsSelected && <text style={{ fg: theme.status.success }}>● Current model</text>}
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
        summary={`${eligible.filter(({ _tag }) => _tag !== "Provider").length} local`}
        hints={eligible.length === 0
          ? "↑↓ choose · Enter open · R refresh · Esc close"
          : "↑↓ choose · Enter select · F favorite · D details · R refresh · Esc close"}
      />
      <scrollbox
        scrollX={false}
        style={{
          flexGrow: 1,
          minHeight: 0,
          rootOptions: { backgroundColor: theme.background.menu },
          wrapperOptions: { border: false, backgroundColor: theme.background.menu },
          viewportOptions: { backgroundColor: theme.background.menu },
          contentOptions: { flexDirection: "column", paddingLeft: 2, paddingRight: 2, paddingTop: 1 },
        }}
      >
        <box style={{ flexDirection: "row", width: "100%" }}>
          <text style={{ fg: theme.text.metadata, width: 2 }}> </text>
          <text style={{ fg: theme.text.metadata, width: 2 }}> </text>
          <text style={{ fg: theme.text.metadata, flexGrow: 1 }}>MODEL</text>
          <text style={{ fg: theme.text.metadata, width: 14 }}>REQUIREMENTS</text>
          <text style={{ fg: theme.text.metadata, width: 9 }}>CONTEXT</text>
          <text style={{ fg: theme.text.metadata, width: 23 }}>STATUS</text>
        </box>
        {eligible.length === 0 ? (
          <box style={{ flexDirection: "column", paddingLeft: 2 }}>
            <text style={{ fg: theme.status.warning, marginLeft: 2 }}>No model is currently available.</text>
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
        ) : eligible.map((entry, index) => {
          const focused = index === cursorIndex
          const active = isSelected(entry)
          const favorite = isFavorite(entry)
          const rowIndex = index
          const status = modelsMenuStatusPresentation(statusFor(entry))
          return (
            <Button
              key={entry.id}
              onClick={() => choose(entry)}
              onMouseOver={() => setCursorId(entry.id)}
              style={{
                flexDirection: "row",
                width: "100%",
                backgroundColor: active
                  ? focused ? theme.text.body : theme.accent
                  : focused
                  ? theme.background.focused
                  : rowIndex % 2 === 0 ? theme.background.menu : theme.background.alternateRow,
              }}
            >
              <text style={{ fg: active ? theme.background.menu : focused ? theme.accent : theme.text.body, width: 2 }}>{active ? "●" : focused ? "›" : " "}</text>
              <text style={{ fg: active ? theme.background.menu : theme.status.warning, width: 2 }}>{favorite ? "★" : " "}</text>
              <text style={{ fg: active ? theme.background.menu : focused ? theme.accent : theme.text.body, flexGrow: 1 }} attributes={active ? TextAttributes.BOLD : TextAttributes.NONE}>{modelsMenuDisplayName(entry)}</text>
              <text style={{ fg: active ? theme.background.menu : theme.text.metadata, width: 14 }}>{requirementFor(entry)}</text>
              <text style={{ fg: active ? theme.background.menu : theme.text.metadata, width: 9 }}>{modelsMenuContextLabel(entry)}</text>
              <text
                style={{
                  fg: active
                    ? theme.background.menu
                    : status.tone === "warning" ? theme.status.warning : theme.text.metadata,
                  width: 23,
                }}
                attributes={active ? TextAttributes.BOLD : TextAttributes.NONE}
                wrapMode="none"
              >
                {truncateToDisplayWidth(status.label, 23)}
              </text>
            </Button>
          )
        })}
        {Result.isFailure(config.catalog) && (
          <text style={{ fg: theme.status.failure }}>Unable to refresh the provider model catalog; showing the last usable state when available.</text>
        )}
        {Result.isFailure(slotActions.assignResult) && (
          <text style={{ fg: theme.status.failure }}>Failed to update model selection.</text>
        )}
        {installationFailed && (
          <text style={{ fg: theme.status.failure }}>Failed to install the local model.</text>
        )}
        {Result.isFailure(config.favoriteUpdate) && (
          <text style={{ fg: theme.status.failure }}>Failed to update model favorite.</text>
        )}
      </scrollbox>
    </>
  )
})

export const huggingFaceRepositoryUrls = (
  model: LocalModel,
): readonly string[] => [...new Set(servableModelBundlePackages(model.bundle).flatMap(({ source }) =>
  source._tag === "HuggingFace"
    ? [`https://huggingface.co/${source.repository}`]
    : [],
))]

export const catalogStatus = (
  model: LocalModel,
  reconciliationState: CatalogModelReconciliationState = { _tag: "Idle" },
): string => {
  if (reconciliationState._tag === "Removing") return "Removing…"
  if (reconciliationState._tag === "RemoveFailed") return "Remove failed"
  if (reconciliationState._tag === "Transferring") {
    const verb = reconciliationState.operation === "Update" ? "Updating" : "Downloading"
    return `${verb} ${Math.round(reconciliationState.completedBytes
      / Math.max(1, reconciliationState.totalBytes) * 100)}%`
  }
  if (reconciliationState._tag === "Starting") {
    return reconciliationState.operation === "Update" ? "Starting update…" : "Starting download…"
  }
  if (reconciliationState._tag === "Failed") {
    return reconciliationState.operation === "Update" ? "Update failed" : "Download failed"
  }
  const acquisitionState = model.acquisitionState
  if (model.upgradeState._tag === "Available") return "Update available"
  if (model.upgradeState._tag === "Failed") return "Update failed"
  if (acquisitionState._tag === "NotInstalled"
    || acquisitionState._tag === "Cancelled") return "Available"
  if (acquisitionState._tag === "Failed") return "Download failed"
  if (model.servingState._tag === "Assessed"
    && model.servingState.availabilityState._tag === "Unavailable") {
    return model.servingState.availabilityState.failure.message
  }
  return localModelInstalledStatus(model)
}

const CatalogCandidateRow = memo(function CatalogCandidateRow({
  model,
  memoryBytes,
  highlighted,
  focused,
  selected,
  pendingDelete,
  reconciliationState,
  index,
  layout,
  rowId,
  onClick,
  onMouseOver,
}: {
  readonly model: LocalModel
  readonly memoryBytes: number | undefined
  readonly highlighted: boolean
  readonly focused: boolean
  readonly selected: boolean
  readonly pendingDelete: boolean
  readonly reconciliationState: CatalogModelReconciliationState
  readonly index: number
  readonly layout: CatalogLayout
  readonly rowId: string
  readonly onClick: () => void
  readonly onMouseOver: () => void
}) {
  const theme = useTheme()
  const status = pendingDelete
    ? "Remove? ↵/Esc"
    : catalogStatus(model, reconciliationState)
  const statusColor = pendingDelete
    ? theme.status.warning
    : model.acquisitionState._tag === "Failed"
      || reconciliationState._tag === "Failed"
      || reconciliationState._tag === "RemoveFailed"
      ? theme.status.failure
      : reconciliationState._tag === "Starting"
        || reconciliationState._tag === "Transferring"
        || reconciliationState._tag === "Removing"
        || model.acquisitionState._tag === "Downloading"
        || model.acquisitionState._tag === "Installed"
        ? theme.accent
        : theme.text.metadata
  const memoryText = memoryBytes === undefined ? "—" : formatBytes(memoryBytes)
  const speedText = performanceRangeSpeedLabel(model, "t/s")
  const speculativeMethod = localModelSpeculativeMethodLabel(model)
  const speculativeText = Option.getOrElse(speculativeMethod, () => "—")
  const backgroundColor = highlighted
    ? theme.background.focused
    : index % 2 === 0 ? theme.background.menu : theme.background.alternateRow

  return (
    <Button
      id={rowId}
      onClick={onClick}
      onMouseOver={onMouseOver}
      style={{
        flexDirection: "row",
        columnGap: layout.columnGap,
        width: "100%",
        height: 1,
        minHeight: 1,
        flexShrink: 0,
        backgroundColor,
      }}
    >
      <text style={{ fg: focused ? theme.accent : theme.text.body, width: 1 }} wrapMode="none">
        {selected ? "●" : focused ? "›" : " "}
      </text>
      <text style={{ fg: focused ? theme.accent : theme.text.body, width: layout.modelWidth }} wrapMode="none">
        {formatCatalogModelLabel(model.presentation.displayName, model.presentation.variantLabel, layout.modelWidth)}
      </text>
      {layout.showMemory && (
        <text style={{ fg: theme.text.metadata, width: layout.columns.memory }} wrapMode="none">
          {truncateToDisplayWidth(memoryText, layout.columns.memory)}
        </text>
      )}
      {layout.showSpeed && (
        <text style={{ fg: theme.text.metadata, width: layout.columns.speed }} wrapMode="none">
          {truncateToDisplayWidth(speedText, layout.columns.speed)}
        </text>
      )}
      {layout.showSpeculative && (
        <text style={{ fg: theme.text.metadata, width: layout.columns.speculative }} wrapMode="none">
          {truncateToDisplayWidth(speculativeText, layout.columns.speculative)}
        </text>
      )}
      <text style={{ fg: statusColor, width: layout.columns.status }} wrapMode="none">
        {truncateToDisplayWidth(status, layout.columns.status)}
      </text>
    </Button>
  )
})

export type CatalogInspectorActionId =
  | "primary"
  | "select"
  | "cancel"
  | "load"
  | "stop"
  | "uninstall"

type CatalogPrimarySlot = ModelSlotsState["slots"]["primary"]

interface CatalogActionHoverTarget {
  readonly configurationId: string
  readonly action: CatalogInspectorActionId
}

const catalogModelIsSelected = (
  model: LocalModel,
  selected: Option.Option<Pick<ProviderModelCatalogEntry, "providerId" | "providerModelId">>,
): boolean => Option.exists(selected, (selection) =>
  selection.providerId === LOCAL_PROVIDER_ID
  && Option.contains(localModelProviderModelId(model), selection.providerModelId))

const catalogSlotForModel = (
  model: LocalModel,
  slot: CatalogPrimarySlot | null,
): CatalogPrimarySlot | null => slot?._tag === "ConfiguredLocal"
  && slot.selection.providerId === LOCAL_PROVIDER_ID
  && Option.contains(localModelProviderModelId(model), slot.selection.providerModelId)
  ? slot
  : null

export const catalogInspectorActions = (
  model: LocalModel,
  reconciliationState: CatalogModelReconciliationState,
  selected = false,
  selectedSlot: CatalogPrimarySlot | null = null,
): readonly CatalogInspectorActionId[] => {
  if (reconciliationState._tag === "Removing") return []
  if (reconciliationState._tag === "Transferring"
    || model.acquisitionState._tag === "Downloading"
    || model.upgradeState._tag === "Upgrading") return ["cancel"]
  if (reconciliationState._tag === "Starting") return []
  if (model.acquisitionState._tag !== "Installed") return ["primary"]

  const actions: CatalogInspectorActionId[] = []
  if (!selected) {
    if (model.servingState._tag === "Assessed"
      && model.servingState.availabilityState._tag === "Selectable") actions.push("select")
  } else if (selectedSlot?._tag === "ConfiguredLocal") {
    if (selectedSlot.actions.some((action) => action === "Load" || action === "RetryLoad")) actions.push("load")
    else if (selectedSlot.actions.includes("Stop")) actions.push("stop")
  }
  if (model.upgradeState._tag === "Available"
    || model.upgradeState._tag === "Failed") actions.push("primary")
  actions.push("uninstall")
  return actions
}

export const catalogInspectorActionLabel = (
  action: CatalogInspectorActionId,
  model: LocalModel,
  selectedSlot: CatalogPrimarySlot | null = null,
  reconciliationState: CatalogModelReconciliationState = { _tag: "Idle" },
): string => {
  switch (action) {
    case "select": return "Select model"
    case "primary": {
      if (reconciliationState._tag === "Failed" && reconciliationState.operation === "Update") {
        return "Retry update"
      }
      if (model.upgradeState._tag === "Available") return "Update"
      if (model.upgradeState._tag === "Failed") return "Retry update"
      const verb = reconciliationState._tag === "Failed" || model.acquisitionState._tag === "Failed"
        ? "Retry download"
        : "Download"
      const totalBytes = model.acquisitionState._tag === "Installed"
        ? model.downloadBytes
        : model.acquisitionState.totalBytes
      return `${verb} (${formatBytes(totalBytes)})`
    }
    case "cancel": return reconciliationState._tag === "Transferring"
      ? reconciliationState.operation === "Update" ? "Cancel update" : "Cancel download"
      : model.upgradeState._tag === "Upgrading" ? "Cancel update" : "Cancel download"
    case "load": return selectedSlot?._tag === "ConfiguredLocal"
      && selectedSlot.actions.includes("RetryLoad") ? "Retry loading" : "Load model"
    case "stop": return selectedSlot?._tag === "ConfiguredLocal"
      && (selectedSlot.residency._tag === "Requested"
        || selectedSlot.residency._tag === "Loading")
      ? "Cancel loading" : "Stop model"
    case "uninstall": return reconciliationState._tag === "RemoveFailed"
      ? "Retry uninstall"
      : "Uninstall"
  }
}

const catalogInspectorStatus = (
  model: LocalModel,
  reconciliationState: CatalogModelReconciliationState,
  selected: boolean,
  selectedSlot: CatalogPrimarySlot | null,
): string => {
  if (reconciliationState._tag === "Removing") return "REMOVING…"
  if (reconciliationState._tag === "RemoveFailed") return "REMOVE FAILED"
  if (reconciliationState._tag === "Transferring") {
    const label = reconciliationState.operation === "Update" ? "UPDATING" : "DOWNLOADING"
    return `${label} ${Math.round(reconciliationState.completedBytes / Math.max(1, reconciliationState.totalBytes) * 100)}%`
  }
  if (reconciliationState._tag === "Starting") return reconciliationState.operation === "Update"
    ? "STARTING UPDATE…" : "STARTING DOWNLOAD…"
  if (reconciliationState._tag === "Failed") return reconciliationState.operation === "Update"
    ? "UPDATE FAILED" : "DOWNLOAD FAILED"
  if (model.acquisitionState._tag === "Downloading") {
    return `DOWNLOADING ${Math.round(model.acquisitionState.completedBytes / Math.max(1, model.acquisitionState.totalBytes) * 100)}%`
  }
  if (model.upgradeState._tag === "Upgrading") {
    return `UPDATING ${Math.round(model.upgradeState.completedBytes / Math.max(1, model.upgradeState.totalBytes) * 100)}%`
  }
  if (model.acquisitionState._tag === "Failed") return "DOWNLOAD FAILED"
  if (selected) {
    const updateSuffix = model.upgradeState._tag === "Available"
      ? " · UPDATE AVAILABLE"
      : model.upgradeState._tag === "Failed" ? " · UPDATE FAILED" : ""
    if (selectedSlot?._tag === "ConfiguredLocal") {
      const residency = selectedSlot.residency
      if (residency._tag === "Requested") return `LOADING 0%${updateSuffix}`
      if (residency._tag === "Loading") return `LOADING ${Math.round(Option.getOrElse(residency.progress, () => 0) * 100)}%${updateSuffix}`
      if (residency._tag === "Ready") return `SELECTED · READY${updateSuffix}`
      if (residency._tag === "Stopping") return "STOPPING…"
      if (residency._tag === "Failed") return "LOAD FAILED"
    }
    return `SELECTED${updateSuffix}`
  }
  return catalogStatus(model, reconciliationState).toUpperCase()
}

const CatalogInspector = memo(function CatalogInspector({
  model,
  reconciliationState,
  selected,
  selectedSlot,
  transition,
  actions,
  actionCursor,
  actionsFocused,
  hoveredAction,
  confirmingUninstall,
  onAction,
  onActionHover,
}: {
  readonly model: LocalModel
  readonly reconciliationState: CatalogModelReconciliationState
  readonly selected: boolean
  readonly selectedSlot: CatalogPrimarySlot | null
  readonly transition: PentagonRadarTransition | null
  readonly actions: readonly CatalogInspectorActionId[]
  readonly actionCursor: number
  readonly actionsFocused: boolean
  readonly hoveredAction: CatalogInspectorActionId | null
  readonly confirmingUninstall: boolean
  readonly onAction: (action: CatalogInspectorActionId) => void
  readonly onActionHover: (action: CatalogInspectorActionId | null) => void
}) {
  const theme = useTheme()
  const platform = usePlatform()
  const [sourceHovered, setSourceHovered] = useState(false)
  const radarAxes = localModelRadarAxes(model)
  const status = catalogInspectorStatus(model, reconciliationState, selected, selectedSlot)
  const contentWidth = CATALOG_INSPECTOR_CONTENT_WIDTH
  const repositoryUrl = huggingFaceRepositoryUrls(model)[0]
  const repository = repositoryUrl?.replace("https://", "")
  const license = Option.getOrElse(model.presentation.license, () => "Unknown license")
  const classification = model.catalogMembershipState._tag === "InCatalog"
    && model.servingState._tag === "Assessed"
    ? formatModelClassification(
        model.catalogMembershipState.catalogData.parameterization,
        model.servingState.capabilities.vision,
      )
    : ""

  return (
    <box style={{ flexGrow: 1, minHeight: 0, width: "100%", flexDirection: "column", paddingLeft: 2, paddingRight: 2 }}>
      <box style={{ height: CATALOG_SPLIT_INSPECTOR_HEIGHTS.identity, minHeight: CATALOG_SPLIT_INSPECTOR_HEIGHTS.identity, flexShrink: 0, flexDirection: "column" }}>
        <box style={{ flexDirection: "row", width: "100%" }}>
          <text style={{ fg: theme.text.body, flexGrow: 1 }} attributes={TextAttributes.BOLD} wrapMode="none">
            {truncateToDisplayWidth(formatLocalModelDisplayName(model), Math.max(1, contentWidth - status.length - 1))}
          </text>
          <text style={{ fg: reconciliationState._tag === "Failed"
            || reconciliationState._tag === "RemoveFailed"
            || model.upgradeState._tag === "Failed" ? theme.status.failure : theme.accent }} wrapMode="none">{status}</text>
        </box>
        <text style={{ fg: theme.text.supporting }} wrapMode="none">
          {truncateToDisplayWidth(classification, contentWidth)}
        </text>
        <text> </text>
      </box>
      <box style={{ width: "100%", height: CATALOG_SPLIT_INSPECTOR_HEIGHTS.metrics, minHeight: CATALOG_SPLIT_INSPECTOR_HEIGHTS.metrics, flexShrink: 0 }}>
        <CatalogRadarView axes={radarAxes} transition={transition} />
      </box>
      <box style={{ height: CATALOG_SPLIT_INSPECTOR_HEIGHTS.info, minHeight: CATALOG_SPLIT_INSPECTOR_HEIGHTS.info, flexShrink: 0, flexDirection: "column" }}>
        <text style={{ fg: theme.text.metadata }} attributes={TextAttributes.BOLD} wrapMode="none">SOURCE</text>
        <text style={{ fg: theme.text.metadata }} wrapMode="none">{truncateToDisplayWidth(`License ${license}`, contentWidth)}</text>
        {repositoryUrl === undefined ? (
          <text style={{ fg: theme.text.metadata }} wrapMode="none">Repository unavailable</text>
        ) : (
          <Button
            onClick={() => { void platform.openLink(repositoryUrl) }}
            onMouseOver={() => setSourceHovered(true)}
            onMouseOut={() => setSourceHovered(false)}
          >
            <text wrapMode="none">
              <span
                fg={sourceHovered ? theme.link : theme.text.metadata}
                attributes={sourceHovered ? TextAttributes.UNDERLINE : TextAttributes.NONE}
              >
                {truncateToDisplayWidth(repository, contentWidth - (sourceHovered ? 1 : 0))}{sourceHovered ? "↗" : ""}
              </span>
            </text>
          </Button>
        )}
      </box>
      <box style={{ flexGrow: 1, minHeight: 1 }} />
      <box style={{
        height: CATALOG_SPLIT_INSPECTOR_HEIGHTS.actions,
        minHeight: CATALOG_SPLIT_INSPECTOR_HEIGHTS.actions,
        flexShrink: 0,
        flexDirection: "column",
      }}>
        <text style={{ fg: theme.text.metadata }} attributes={TextAttributes.BOLD} wrapMode="none">ACTIONS</text>
        <box style={{ flexDirection: "column" }}>
          {actions.length === 0 ? (
            <text style={{ fg: theme.text.disabled }}>
              {reconciliationState._tag === "Removing" ? "Removing…" : "No actions available"}
            </text>
          ) : actions.map((action, index) => {
            const focused = hoveredAction === action || (actionsFocused && index === actionCursor)
            return action === "uninstall" && confirmingUninstall ? (
              <text key={action} wrapMode="none">
                <span fg={focused ? theme.accent : theme.text.body}>{focused ? "› " : "  "}Remove model? </span>
                <span fg={theme.text.metadata}>[Enter] confirm · [Esc] cancel</span>
              </text>
            ) : (
              <MenuAction
                key={action}
                label={catalogInspectorActionLabel(action, model, selectedSlot, reconciliationState)}
                focused={focused}
                tone="normal"
                onClick={() => onAction(action)}
                onMouseOver={() => {
                  onActionHover(action)
                }}
                onMouseOut={() => onActionHover(null)}
              />
            )
          })}
        </box>
      </box>
    </box>
  )
})

const CatalogMenu = memo(function CatalogMenu({
  initialCatalogDetailId,
  setRootSwitchingEnabled,
}: CatalogMenuProps) {
  const theme = useTheme()
  const config = useModelConfig()
  const menuSize = useLocalWidth()
  const catalogScrollRef = useRef<ScrollBoxRenderable | null>(null)
  const menuWidth = menuSize.width ?? 80
  const layout = deriveCatalogLayout(menuWidth)
  const catalogView = Result.value(useCatalogModels())
  const modelActions = useLocalModelActions()
  const slotActions = useModelSlotActions()
  const catalogModels = Option.match(catalogView, {
    onNone: () => [],
    onSome: ({ models }) => models.filter(({ model }) =>
      model.servingState._tag === "Assessed"
        && model.servingState.assessment._tag === "Fits"),
  })
  const recommendationsReady = Option.exists(
    catalogView,
    (models) => models.discoveryState._tag === "Ready",
  )
  const memoryBytesFor = (model: LocalModel): number | undefined =>
    model.servingState._tag === "Assessed"
      && model.servingState.assessment._tag === "Fits"
      ? model.servingState.assessment.memory.totalRequiredBytes
      : undefined
  const candidates = catalogModels.map(({ model }) => model).sort((left, right) => {
    const leftInstalled = left.acquisitionState._tag === "Installed"
    const rightInstalled = right.acquisitionState._tag === "Installed"
    return (leftInstalled === rightInstalled ? 0 : leftInstalled ? -1 : 1)
      || left.presentation.displayName.localeCompare(right.presentation.displayName)
      || left.presentation.variantLabel.localeCompare(right.presentation.variantLabel)
      || Option.getOrElse(localModelConfigurationId(left), () => "")
        .localeCompare(Option.getOrElse(localModelConfigurationId(right), () => ""))
  })
  const [cursorId, setCursorId] = useState<string | null>(null)
  const [detailId, setDetailId] = useState<string | null>(initialCatalogDetailId)
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null)
  const [radarTransition, setRadarTransition] = useState<PentagonRadarTransition | null>(null)
  const [actionHoverTarget, setActionHoverTarget] = useState<CatalogActionHoverTarget | null>(null)
  const configurationIdFor = (model: LocalModel) => Option.getOrUndefined(
    localModelConfigurationId(model),
  )
  const reconciliationStateFor = (model: LocalModel): CatalogModelReconciliationState => {
    const configurationId = configurationIdFor(model)
    if (configurationId === undefined) return { _tag: "Idle" }
    return catalogModels.find(({ model: candidate }) =>
      configurationIdFor(candidate) === configurationId)?.reconciliationState ?? { _tag: "Idle" }
  }
  const cursorIndex = Math.max(0, candidates.findIndex((model) =>
    configurationIdFor(model) === cursorId))
  const cursor = candidates[cursorIndex]
  const detail = candidates.find((model) => configurationIdFor(model) === detailId) ?? null
  const inspected = detail ?? cursor ?? null
  const inspectedConfigurationId = inspected === null ? undefined : configurationIdFor(inspected)
  const confirmingUninstall = inspectedConfigurationId !== undefined
    && pendingDeleteId === inspectedConfigurationId
  const hoveredInspectorAction = inspectedConfigurationId !== undefined
    && actionHoverTarget?.configurationId === inspectedConfigurationId
    ? actionHoverTarget.action
    : null
  const selectedModel = Option.flatMap(config.selections, ({ primary }) => primary)
  const primarySlot = Option.match(Result.value(config.slots), {
    onNone: () => null,
    onSome: ({ state }) => state.slots.primary,
  })
  const inspectedSelected = inspected !== null && catalogModelIsSelected(inspected, selectedModel)
  const inspectedSlot = inspected === null ? null : catalogSlotForModel(inspected, primarySlot)
  const inspectedReconciliationState = inspected === null
    ? { _tag: "Idle" } as const
    : reconciliationStateFor(inspected)
  const progress = Option.match(catalogView, {
    onNone: () => [],
    onSome: (models) => localInferenceProgressLines(models.discoveryState.progress),
  })
  const runningProgress = progress.find((line) => line.state === "running")
  const spinner = useSpinnerFrame(runningProgress !== undefined)
  const inspectorActions = inspected === null
    ? []
    : catalogInspectorActions(inspected, inspectedReconciliationState, inspectedSelected, inspectedSlot)
  const inspectorActionCursor = useBoundedCursor(inspectorActions.length)
  const moveCursorTo = useCallback((index: number) => {
    const model = candidates[index]
    const configurationId = model && configurationIdFor(model)
    if (!model || configurationId === undefined) return
    setActionHoverTarget(null)
    const fromAxes = cursor === undefined ? Option.none() : localModelRadarAxes(cursor)
    const toAxes = localModelRadarAxes(model)
    if (Option.isSome(fromAxes) && Option.isSome(toAxes)
      && configurationIdFor(cursor!) !== configurationId) {
      const now = getAnimationTimeSnapshot()
      setRadarTransition(retargetPentagonRadar(
        pentagonRadarValues(fromAxes.value),
        pentagonRadarValues(toAxes.value),
        radarTransition,
        now,
      ))
    } else {
      setRadarTransition(null)
    }
    setCursorId(configurationId)
    scrollCatalogCandidateIntoView(catalogScrollRef.current, configurationId)
  }, [candidates, cursor, radarTransition])

  const primaryAction = useCallback((model: LocalModel) => {
    const configurationId = configurationIdFor(model)
    if (configurationId === undefined
      || model.acquisitionState._tag === "Downloading"
      || (model.acquisitionState._tag === "Installed"
        && model.upgradeState._tag !== "Available"
        && model.upgradeState._tag !== "Failed")
      || reconciliationStateFor(model)._tag === "Starting") return
    modelActions.install(configurationId)
  }, [modelActions, catalogModels])

  const selectCandidate = useCallback((model: LocalModel) => {
    if (model.servingState._tag !== "Assessed"
      || model.servingState.assessment._tag !== "Fits"
      || model.servingState.availabilityState._tag !== "Selectable") return
    const reasoningEffort = model.servingState.capabilities.reasoning.defaultEffort
    const providerModelId = localModelProviderModelId(model)
    const assign = (id: ProviderModelId) => slotActions.assign(PRIMARY_SLOT_ID, {
      providerId: LOCAL_PROVIDER_ID,
      providerModelId: id,
      reasoningEffort: Option.getOrElse(
        reasoningEffort,
        () => ReasoningEffortSchema.make("none"),
      ),
    })
    if (Option.isSome(providerModelId)) {
      void assign(providerModelId.value)
    }
  }, [slotActions])

  const runInspectorAction = useCallback((action: CatalogInspectorActionId) => {
    if (inspected === null) return
    if (action !== "uninstall") setPendingDeleteId(null)
    if (action === "primary") {
      primaryAction(inspected)
      return
    }
    if (action === "select") {
      selectCandidate(inspected)
      return
    }
    if (action === "cancel") {
      if (inspectedReconciliationState._tag === "Transferring") {
        modelActions.cancel(inspectedReconciliationState.downloadId)
      } else if (inspected.acquisitionState._tag === "Downloading") {
        modelActions.cancel(inspected.acquisitionState.downloadId)
      } else if (inspected.upgradeState._tag === "Upgrading") {
        modelActions.cancel(inspected.upgradeState.downloadId)
      }
      return
    }
    if (action === "load" && inspectedSlot?._tag === "ConfiguredLocal") {
      void slotActions.load(PRIMARY_SLOT_ID)
      return
    }
    if (action === "stop" && inspectedSlot?._tag === "ConfiguredLocal") {
      void slotActions.stop(PRIMARY_SLOT_ID)
      return
    }
    if (action === "uninstall" && inspected.acquisitionState._tag === "Installed") {
      const configurationId = configurationIdFor(inspected)
      if (configurationId !== undefined) setPendingDeleteId(configurationId)
    }
  }, [inspected, inspectedReconciliationState, inspectedSlot, modelActions, primaryAction, selectCandidate, slotActions])

  useKeyboard(useCallback((key: KeyEvent) => {
    if (key.defaultPrevented) return
    if (confirmingUninstall && pendingDeleteId !== null) {
      if (key.name === "return" || key.name === "enter") {
        const model = candidates.find((candidate) => configurationIdFor(candidate) === pendingDeleteId)
        const configurationId = model && Option.getOrUndefined(localModelConfigurationId(model))
        if (model?.acquisitionState._tag === "Installed" && configurationId !== undefined) {
          modelActions.delete(configurationId)
        }
        setPendingDeleteId(null)
        key.preventDefault()
        return
      } else if (key.name === "escape") {
        setPendingDeleteId(null)
        key.preventDefault()
        return
      } else if (key.name === "up" || key.name === "down" || key.name === "k" || key.name === "j") {
        setPendingDeleteId(null)
      } else {
        return
      }
    }
    if (detail) {
      if (key.name === "escape") {
        key.preventDefault()
        setDetailId(null)
        setRadarTransition(null)
        setRootSwitchingEnabled(true)
      } else if (key.name === "up" && inspectorActions.length > 0) {
        key.preventDefault()
        inspectorActionCursor.previous()
      } else if (key.name === "down" && inspectorActions.length > 0) {
        key.preventDefault()
        inspectorActionCursor.next()
      } else if (key.name === "return" || key.name === "enter") {
        const action = inspectorActions[inspectorActionCursor.index]
        if (action !== undefined) {
          key.preventDefault()
          runInspectorAction(action)
        }
      }
      return
    }
    setActionHoverTarget(null)
    if (pendingDeleteId !== null) {
      const confirmsDelete = key.name === "return" || key.name === "enter"
      if (confirmsDelete) {
        const model = candidates.find((candidate) => configurationIdFor(candidate) === pendingDeleteId)
        const configurationId = model && Option.getOrUndefined(localModelConfigurationId(model))
        if (model?.acquisitionState._tag === "Installed" && configurationId !== undefined) {
          modelActions.delete(configurationId)
        }
        setPendingDeleteId(null)
        key.preventDefault()
        return
      }
      setPendingDeleteId(null)
      if (key.name === "escape" || key.name === "backspace") {
        key.preventDefault()
        return
      }
    }
    if ((key.name === "up" || key.name === "k") && candidates.length > 0) {
      key.preventDefault()
      moveCursorTo(Math.max(0, cursorIndex - 1))
    } else if ((key.name === "down" || key.name === "j") && candidates.length > 0) {
      key.preventDefault()
      moveCursorTo(Math.min(candidates.length - 1, cursorIndex + 1))
    } else if ((key.name === "return" || key.name === "enter") && cursor) {
      key.preventDefault()
      setRadarTransition(null)
      inspectorActionCursor.reset()
      setDetailId(configurationIdFor(cursor) ?? null)
      setRootSwitchingEnabled(false)
    } else if (key.name === "d" && cursor) {
      key.preventDefault()
      primaryAction(cursor)
    } else if (key.name === "s" && cursor && cursor.servingState._tag === "Assessed"
      && cursor.servingState.availabilityState._tag === "Selectable") {
      key.preventDefault()
      selectCandidate(cursor)
    } else if (key.name === "backspace" && cursor) {
      if (cursor.acquisitionState._tag === "Downloading") {
        modelActions.cancel(cursor.acquisitionState.downloadId)
        key.preventDefault()
      } else if (cursor.upgradeState._tag === "Upgrading") {
        modelActions.cancel(cursor.upgradeState.downloadId)
        key.preventDefault()
      } else if (cursor.acquisitionState._tag === "Installed") {
        setPendingDeleteId(configurationIdFor(cursor) ?? null)
        key.preventDefault()
      }
    }
  }, [candidates, confirmingUninstall, cursor, cursorIndex, detail, inspectorActionCursor, inspectorActions, modelActions, moveCursorTo, pendingDeleteId, primaryAction, runInspectorAction, selectCandidate, setRootSwitchingEnabled]))

  if (menuSize.width === null) {
    return (
      <box
        ref={menuSize.ref}
        onSizeChange={menuSize.onSizeChange}
        style={{ flexGrow: 1, minHeight: 0, flexDirection: "column" }}
      />
    )
  }

  const inspectorView = inspected === null ? null : (
    <>
      <CatalogInspector
        key={inspectedConfigurationId}
        model={inspected}
        reconciliationState={inspectedReconciliationState}
        selected={inspectedSelected}
        selectedSlot={inspectedSlot}
        transition={detail === null ? radarTransition : null}
        actions={inspectorActions}
        actionCursor={inspectorActionCursor.index}
        actionsFocused={detail !== null}
        hoveredAction={hoveredInspectorAction}
        confirmingUninstall={confirmingUninstall}
        onAction={runInspectorAction}
        onActionHover={(action) => {
          setActionHoverTarget(action === null || inspectedConfigurationId === undefined
            ? null
            : { configurationId: inspectedConfigurationId, action })
        }}
      />
      {Result.isFailure(slotActions.assignResult) && (
        <text style={{ fg: theme.status.failure }}>Failed to update model selection.</text>
      )}
    </>
  )

  if (detail && layout.mode !== "split") {
    return (
      <box
        ref={menuSize.ref}
        onSizeChange={menuSize.onSizeChange}
        style={{ flexGrow: 1, minHeight: 0, flexDirection: "column" }}
      >
        <MenuHeader
          title="Catalog"
          selection={formatLocalModelDisplayName(detail)}
          onSectionClick={() => {
            setDetailId(null)
            setRadarTransition(null)
            setRootSwitchingEnabled(true)
          }}
          hints={catalogDetailHints(layout.compactHeader)}
          compact={layout.compactHeader}
          width={menuWidth}
        />
        <box style={{ flexGrow: 1, minHeight: 0, flexDirection: "column", paddingTop: 1 }}>
          {inspectorView}
        </box>
      </box>
    )
  }

  const list = (
    <scrollbox ref={catalogScrollRef} scrollX={false} onMouseOver={() => setActionHoverTarget(null)} style={{
      width: layout.mode === "split" ? layout.listWidth : "100%",
      flexGrow: layout.mode === "split" ? 0 : 1,
      minHeight: 0,
      rootOptions: { backgroundColor: theme.background.menu },
      wrapperOptions: { border: false, backgroundColor: theme.background.menu },
      viewportOptions: { backgroundColor: theme.background.menu },
      contentOptions: { flexDirection: "column", paddingLeft: 2, paddingRight: 2 },
    }}>
      <box style={{
        flexDirection: "row",
        columnGap: layout.columnGap,
        width: "100%",
        height: 1,
        minHeight: 1,
        flexShrink: 0,
      }}>
        <text style={{ fg: theme.text.metadata, width: 1 }} wrapMode="none"> </text>
        <text style={{ fg: theme.text.metadata, width: layout.modelWidth }} wrapMode="none">MODEL</text>
        {layout.showMemory && <text style={{ fg: theme.text.metadata, width: layout.columns.memory }} wrapMode="none">MEMORY</text>}
        {layout.showSpeed && <text style={{ fg: theme.text.metadata, width: layout.columns.speed }} wrapMode="none">SPEED</text>}
        {layout.showSpeculative && <text style={{ fg: theme.text.metadata, width: layout.columns.speculative }} wrapMode="none">SPECULATIVE</text>}
        <text style={{ fg: theme.text.metadata, width: layout.columns.status }} wrapMode="none">STATUS</text>
      </box>
      {runningProgress && (
        <text style={{ fg: theme.accent, marginLeft: 2 }}>
          {spinner} {runningProgress.label}{runningProgress.metadata}
        </text>
      )}
      {candidates.length === 0 && recommendationsReady ? (
        <text style={{ fg: theme.status.warning, marginLeft: 2 }}>
          No compatible recommended models are currently available.
        </text>
      ) : candidates.map((candidate, index) => {
        const configurationId = configurationIdFor(candidate)
        if (configurationId === undefined) return null
        const highlighted = index === cursorIndex
        return (
          <CatalogCandidateRow
            key={configurationId}
            model={candidate}
            memoryBytes={memoryBytesFor(candidate)}
            highlighted={highlighted}
            focused={highlighted && detail === null}
            selected={catalogModelIsSelected(candidate, selectedModel)}
            pendingDelete={pendingDeleteId === configurationId}
            reconciliationState={reconciliationStateFor(candidate)}
            index={index}
            layout={layout}
            rowId={catalogCandidateRowId(configurationId)}
            onClick={() => {
              setPendingDeleteId(null)
              moveCursorTo(index)
              setRadarTransition(null)
              inspectorActionCursor.reset()
              setDetailId(configurationId)
              setRootSwitchingEnabled(false)
            }}
            onMouseOver={() => {
              if (detail !== null && detailId === configurationId) return
              if (detail !== null) {
                setDetailId(null)
                setRootSwitchingEnabled(true)
                inspectorActionCursor.reset()
              }
              moveCursorTo(index)
              if (pendingDeleteId !== configurationId) setPendingDeleteId(null)
            }}
          />
        )
      })}
    </scrollbox>
  )

  return (
    <box
      ref={menuSize.ref}
      onSizeChange={menuSize.onSizeChange}
      style={{ flexGrow: 1, minHeight: 0, flexDirection: "column" }}
    >
      <MenuHeader
        title="Catalog"
        selection={detail === null ? undefined : formatLocalModelDisplayName(detail)}
        onSectionClick={detail === null ? undefined : () => {
          setDetailId(null)
          setRadarTransition(null)
          setRootSwitchingEnabled(true)
        }}
        subtitle={detail !== null || layout.compactHeader ? undefined : "Find and download local models"}
        summary={detail !== null ? undefined
          : layout.compactHeader ? String(candidates.length) : `${candidates.length} compatible`}
        hints={detail === null
          ? catalogListHints(menuWidth)
          : catalogDetailHints(layout.compactHeader)}
        compact={layout.compactHeader}
        width={menuWidth}
      />
      <box style={{ flexGrow: 1, minHeight: 0, flexDirection: "row", paddingTop: 1 }}>
        {list}
        {layout.mode === "split" && inspected !== null && (
          <>
            <box style={{ width: layout.dividerWidth, backgroundColor: theme.border.standard }} />
            <box style={{ width: layout.inspectorWidth, minWidth: layout.inspectorWidth, flexDirection: "column" }}>
              {inspectorView}
            </box>
          </>
        )}
      </box>
    </box>
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
    void slotActions.stop(PRIMARY_SLOT_ID)
  }, [action, slotActions])

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
          rootOptions: { backgroundColor: theme.background.menu },
          wrapperOptions: { border: false, backgroundColor: theme.background.menu },
          viewportOptions: { backgroundColor: theme.background.menu },
          contentOptions: { flexDirection: "column", paddingLeft: 2, paddingRight: 2, paddingTop: 1 },
        }}
      >
        {Option.match(hardwareSnapshot, {
          onNone: () => (
            <text style={{ fg: Result.isFailure(hardwareState) ? theme.status.failure : theme.text.metadata }}>
              {Result.isFailure(hardwareState) ? "Hardware detection is unavailable." : "Detecting local-inference hardware…"}
            </text>
          ),
          onSome: (detectedHardware) => {
            const hardware = describeLocalHardware(detectedHardware)
            return (
              <>
                <text style={{ fg: theme.text.body }} attributes={TextAttributes.BOLD}>{hardware.system.name}</text>
                {hardware.system.details.map((line) => <text key={line} style={{ fg: theme.text.metadata }}>{line}</text>)}
                {hardware.accelerators.map((accelerator) => (
                  <text key={`${accelerator.name}:${accelerator.details}`} style={{ fg: theme.text.metadata }}>{accelerator.name} · {accelerator.details}</text>
                ))}
                {hardware.accelerators.length === 0 && !detectedHardware.memoryDomains.some((domain) => domain.kind === "UnifiedMemory") && (
                  <text style={{ fg: theme.text.metadata }}>CPU inference · No GPU detected</text>
                )}
              </>
            )
          },
        })}
        <box style={{ flexDirection: "column", marginTop: 1 }}>
          <text style={{ fg: theme.text.metadata }} attributes={TextAttributes.BOLD}>CURRENT MODEL</text>
          {currentModel._tag === "NoSelection"
            ? <text style={{ fg: theme.text.metadata }}>No local model selected</text>
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
                    <text style={{ fg: theme.text.body, flexGrow: 1 }} attributes={TextAttributes.BOLD}>{currentModel.displayName}</text>
                    <text style={{ fg: currentModel._tag === "Running" ? theme.accent : theme.text.metadata }}>{status}</text>
                  </box>
                  {currentModel._tag === "NotLoaded" || currentModel._tag === "Failed"
                    ? <ModelLoadPlanDetails />
                    : (
                        <box style={{ flexDirection: "row" }}>
                          <text style={{ fg: theme.text.metadata, width: 20 }}>Context window</text>
                          <text style={{ fg: theme.text.body, width: 16 }}>
                            {Option.match(currentModel.contextWindow, {
                              onNone: () => "—",
                              onSome: (tokens) => `${formatContextWindow(tokens)} tokens`,
                            })}
                          </text>
                          <text style={{ fg: theme.text.metadata, width: 16 }}>Parallelism</text>
                          <text style={{ fg: theme.text.body }}>
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
            <text style={{ fg: theme.text.metadata }} attributes={TextAttributes.BOLD}>ACTIONS</text>
            {Option.match(action, {
              onNone: () =>
                currentSlot.value.residency._tag === "Stopping"
                  ? <text style={{ fg: theme.text.metadata }}>Stopping model…</text>
                  : currentSlot.value.availability._tag === "Pending"
                    ? <text style={{ fg: theme.text.metadata }}>Initializing model…</text>
                    : currentSlot.value.availability._tag === "Unavailable"
                      ? <text style={{ fg: theme.text.metadata }}>{currentSlot.value.availability.failure.message}</text>
                      : <text style={{ fg: theme.text.metadata }}>{"  "}Load model</text>,
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
      <text style={{ fg: theme.text.metadata, width: 20 }}>Context window</text>
      <text style={{ fg: theme.text.body, width: 16 }}>
        {Option.match(plan, {
          onNone: () => "—",
          onSome: ({ contextWindowTokens }) =>
            `${formatContextWindow(contextWindowTokens)} tokens`,
        })}
      </text>
      <text style={{ fg: theme.text.metadata, width: 16 }}>Parallelism</text>
      <text style={{ fg: theme.text.body }}>
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
          <text style={{ fg: theme.text.body }}>API key</text>
          <box style={{ borderStyle: "single", borderColor: error ? theme.status.failure : theme.accent, paddingLeft: 1, paddingRight: 1, width: "100%" }}>
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
          {error && <text style={{ fg: theme.status.failure }}>{error}</text>}
          <text style={{ fg: theme.text.metadata }}>{auth.saving ? "Saving…" : "Enter to save"}</text>
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
          <text style={{ fg: theme.text.body }}>Disconnect Magnitude Cloud?</text>
          <text style={{ fg: theme.text.supporting }}>Cloud models will no longer be available in Models.</text>
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
          <text style={{ fg: theme.text.supporting }}>Magnitude Cloud provides hosted models and hosted research features.</text>
        )}
        {auth.source === "config" && (
          <text style={{ fg: theme.status.success }}>● Connected via API key {auth.maskedKey ? `(${auth.maskedKey})` : ""}</text>
        )}
        {auth.source === "env" && (
          <>
            <text style={{ fg: theme.status.success }}>● Connected via {auth.envVarName}</text>
            <text style={{ fg: theme.text.supporting }}>This key is managed by the environment. Update it and relaunch to change it.</text>
          </>
        )}
        <box style={{ flexDirection: "column", paddingTop: 1 }}>
          {auth.source === "none" && (
            <Button
              onClick={() => runAction("add")}
              onMouseOver={() => actionCursor.select(actionIds.indexOf("add"))}
            >
              <text style={{ fg: theme.accent }}>{selectedAction === "add" ? "› " : "  "}Add API key</text>
            </Button>
          )}
          {auth.source === "config" && (
            <>
              <Button
                onClick={() => runAction("update")}
                onMouseOver={() => actionCursor.select(actionIds.indexOf("update"))}
              >
                <text style={{ fg: selectedAction === "update" ? theme.accent : theme.text.body }}>
                  {selectedAction === "update" ? "› " : "  "}Update API key
                </text>
              </Button>
              <Button
                onClick={() => runAction("disconnect")}
                onMouseOver={() => actionCursor.select(actionIds.indexOf("disconnect"))}
              >
                <text style={{ fg: selectedAction === "disconnect" ? theme.accent : theme.text.body }}>
                  {selectedAction === "disconnect" ? "› " : "  "}Disconnect
                </text>
              </Button>
            </>
          )}
          <box style={{ flexDirection: "row" }}>
            <text style={{ fg: theme.accent }}>{selectedAction === "link" ? "› " : "  "}</text>
            <Button
              onClick={() => runAction("link")}
              onMouseOver={() => actionCursor.select(actionIds.indexOf("link"))}
            >
              <text style={{ fg: theme.text.body }}>
                View dashboard{" "}
                <span
                  style={{ fg: selectedAction === "link" ? theme.link : theme.accent }}
                  attributes={TextAttributes.UNDERLINE}
                >
                  {MAGNITUDE_CLOUD_URL}↗
                </span>
              </text>
            </Button>
          </box>
        </box>
        {auth.error && <text style={{ fg: theme.status.failure }}>{auth.error}</text>}
        {connected && cloudModels.length > 0 && (
          <box style={{ flexDirection: "column", paddingTop: 1 }}>
            <text style={{ fg: theme.text.metadata }}>AVAILABLE MODELS</text>
            {cloudModels.map((model) => (
              <text key={providerModelKey(model)} style={{ fg: theme.text.body }}>
                {formatModelDisplayName(model.displayName, model.variantLabel)}<span style={{ fg: theme.text.metadata }}> · {formatContextWindow(model.contextWindow)} context</span>
              </text>
            ))}
          </box>
        )}
      </box>
    </>
  )
})
