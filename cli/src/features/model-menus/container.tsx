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
  modelSlotInstanceId,
  modelSlotResidentAllocation,
  getDisplayWidth,
  localModelConfigurationId,
  localModelProviderModelId,
  localModelCapabilities,
  localModelInstallationIsPending,
  latestLocalModelInstallationMutationState,
  selectedSlotModel,
  truncateToDisplayWidth,
  type NotificationState,
  usePlatform,
  useLocalInferenceHardware,
  useLocalModelActions,
  useLocalModelsSelector,
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
  type LocalModelRecommendation,
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
  localModelMaximumContextLength,
  localModelBundleKey,
  localInferenceProgressLines,
  performanceRangeSpeedLabel,
} from "../local-inference/view-model"
import { deriveSettingsAuthInfo } from "../overlays/auth-display"
import {
  catalogDetailHints,
  catalogListHints,
  deriveCatalogLayout,
  formatCatalogModelLabel,
  type CatalogLayout,
} from "./catalog-layout"
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
  ? installedOriginStatus(model.acquisitionState.origins)
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
  ? entry.model.presentation.displayName
  : entry._tag === "LocalStatus"
    ? entry.model.presentation.displayName
    : entry.model.displayName

const modelsMenuContextLength = (entry: ModelsMenuEntry): number => entry._tag === "Local"
  ? entry.model.servingState._tag === "Assessed"
    ? entry.model.servingState.configuration.profile.contextLength
    : localModelMaximumContextLength(entry.model)
  : entry._tag === "Provider"
    ? entry.model.contextWindow
    : localModelMaximumContextLength(entry.model)

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
        backgroundColor: "transparent",
      }}
    >
      <box style={{ flexGrow: 1, minHeight: 0, flexDirection: "column", backgroundColor: theme.menuBg }}>
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
        <text style={{ fg: theme.muted }}>
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
      style={{ fg: theme.foreground }}
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
        {!compact && subtitle && <text style={{ fg: theme.muted }}> · {subtitle}</text>}
        {!compact && selection && <text style={{ fg: theme.foreground }}> → {selection}</text>}
        <box style={{ flexGrow: 1 }} />
        {summary && <text style={{ fg: theme.muted }}>{summary}</text>}
      </box>
      {compact && selection && (
        <text style={{ fg: theme.foreground }} wrapMode="none">
          {truncateToDisplayWidth(selection, compactSelectionWidth)}
        </text>
      )}
      {displayedHints && <text style={{ fg: theme.muted }} wrapMode="none">{displayedHints}</text>}
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
  const localSnapshot = useLocalModelsSelector(
    selectModelMenusLocalModelsState,
    modelMenusLocalModelsStateEquivalent,
  )
  const modelActions = useLocalModelActions()
  const slotActions = useModelSlotActions()
  const catalog = catalogContents(config)
  const catalogSnapshot = Result.value(config.catalog)
  const slotsSnapshot = Result.value(config.slots)
  const selected = Option.flatMap(
    Option.all({ catalog: catalogSnapshot, slots: slotsSnapshot }),
    ({ catalog, slots }) => selectedSlotModel(catalog.state, slots.state, PRIMARY_SLOT_ID),
  )
  const selectedModel = Option.map(selected, ({ model }) => model)
  const currentRecentModelKeys = Option.match(slotsSnapshot, {
    onNone: () => [] as readonly string[],
    onSome: ({ state }) => state.recentModels.primary.map(providerModelKey),
  })
  const currentFavoriteKeys = new Set(config.favoriteModels.map(providerModelKey))
  const [ordering] = useState(() => ({
    selectedModel,
    recentModelKeys: currentRecentModelKeys,
    favoriteKeys: currentFavoriteKeys,
  }))
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
    const providerModel = modelsMenuProviderModel(entry)
    if (entry._tag === "Provider") {
      return providerModel !== undefined
        && providerModel.supportedSlots.includes(PRIMARY_SLOT_ID)
        && (providerModel.availability._tag === "Available" || isSelected(entry))
    }
    return true
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
  const detailCatalogModel = detail === null ? undefined : modelsMenuLocalModel(detail)
  const detailCatalogConfigurationId = detailCatalogModel?.servingState._tag === "Assessed"
    && detailCatalogModel.catalogMembershipState._tag === "InCatalog"
    ? detailCatalogModel.servingState.configuration.id
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
  const latestInstallationMutationState = modelActions.installationMutationStates.at(-1)
  const installationFailed = latestInstallationMutationState !== undefined
    && Result.isFailure(latestInstallationMutationState.result)

  const statusFor = (entry: ModelsMenuEntry): string => {
    const selectedEntry = isSelected(entry)
    if (entry._tag === "LocalStatus") {
      return localModelReadinessStatus(entry.model)
    }
    if (entry._tag === "Local") {
      const memory = memoryFor(entry)
      if (selectedEntry
        && primarySlot?._tag === "ConfiguredLocal"
        && Option.isSome(primarySlot.instance)
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
    if (localModelInstallationIsPending(
      modelActions.installationMutationStates,
      createAction.configurationId,
    )) return
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
    else if (action === "load") void slotActions.load(PRIMARY_SLOT_ID)
    else if (action === "stop" && primarySlot) {
      Option.match(modelSlotInstanceId(primarySlot), {
        onNone: () => {},
        onSome: slotActions.stop,
      })
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
          <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>
            {detailFavorite ? "★ " : ""}{modelsMenuDisplayName(detail)}
          </text>
          <text style={{ fg: theme.muted }}>
            {detail._tag === "Provider" ? providerKindLabel(detail.provider.kind) : "Local"} · {formatContextWindow(modelsMenuContextLength(detail))} context · {statusFor(detail)}
          </text>
          {detailIsLocal && Option.exists(detailMemory, ({ currentHeadroomState, systemUseState }) =>
            currentHeadroomState._tag === "Insufficient" || systemUseState._tag === "High") && (
            <text style={{ fg: theme.warning }}>
              {Option.exists(detailMemory, ({ currentHeadroomState }) =>
                currentHeadroomState._tag === "Insufficient")
                ? "Not enough free memory right now"
                : "Uses more than the recommended share of system memory"}
            </text>
          )}
          {detailCapabilities && (
            <text style={{ fg: theme.muted }}>
              {detailCapabilities.vision ? "Vision" : "No vision"} · Tools · {detailCapabilities.reasoning.supported ? "Reasoning" : "No reasoning"}
            </text>
          )}
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
                  ? focused ? theme.foreground : theme.primary
                  : focused
                  ? theme.surfaceHover
                  : rowIndex % 2 === 0 ? theme.menuBg : theme.menuAltBg,
              }}
            >
              <text style={{ fg: active ? theme.menuBg : focused ? theme.primary : theme.foreground, width: 2 }}>{active ? "●" : focused ? "›" : " "}</text>
              <text style={{ fg: active ? theme.menuBg : theme.warning, width: 2 }}>{favorite ? "★" : " "}</text>
              <text style={{ fg: active ? theme.menuBg : focused ? theme.primary : theme.foreground, flexGrow: 1 }} attributes={active ? TextAttributes.BOLD : TextAttributes.NONE}>{modelsMenuDisplayName(entry)}</text>
              <text style={{ fg: active ? theme.menuBg : theme.muted, width: 14 }}>{requirementFor(entry)}</text>
              <text style={{ fg: active ? theme.menuBg : theme.muted, width: 9 }}>{formatContextWindow(modelsMenuContextLength(entry))}</text>
              <text
                style={{
                  fg: active
                    ? theme.menuBg
                    : status.tone === "warning" ? theme.warning : theme.muted,
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
          <text style={{ fg: theme.error }}>Unable to refresh the provider model catalog; showing the last usable state when available.</text>
        )}
        {Result.isFailure(slotActions.assignResult) && (
          <text style={{ fg: theme.error }}>Failed to update model selection.</text>
        )}
        {installationFailed && (
          <text style={{ fg: theme.error }}>Failed to install the local model.</text>
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

const intelligenceLabel = (model: LocalModel): string =>
  model.catalogMembershipState._tag === "InCatalog"
    ? `${Math.round(model.catalogMembershipState.catalogData.intelligenceScore)}/100`
    : "—"

const qualityLabel = (model: LocalModel): string =>
  model.catalogMembershipState._tag !== "InCatalog" ? "—"
    : model.catalogMembershipState.catalogData.fidelityRank >= 75
      ? "Near original"
      : model.catalogMembershipState.catalogData.fidelityRank >= 55
        ? "Very high"
        : model.catalogMembershipState.catalogData.fidelityRank >= 45 ? "High" : "Reduced"

const catalogDataLabel = (model: LocalModel): string =>
  model.catalogMembershipState._tag !== "InCatalog"
    ? "not in catalog"
    : `intelligence ${intelligenceLabel(model)} · ${qualityLabel(model)}`

export const huggingFaceRepositoryUrls = (
  model: LocalModel,
): readonly string[] => [...new Set((model.bundle._tag === "Standalone"
  ? [model.bundle.package]
  : [model.bundle.target, model.bundle.draft]).flatMap(({ source }) =>
  source._tag === "HuggingFace"
    ? [`https://huggingface.co/${source.repository}`]
    : [],
))]

export const catalogStatus = (
  model: LocalModel,
  installationStarting = false,
): string => {
  const acquisitionState = model.acquisitionState
  if (acquisitionState._tag === "Downloading") {
    return `Downloading ${Math.round(acquisitionState.completedBytes / Math.max(1, acquisitionState.totalBytes) * 100)}%`
  }
  if (installationStarting) return "Starting download…"
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
  recommendation,
  focused,
  pendingDelete,
  installationStarting,
  index,
  layout,
  rowId,
  onClick,
  onMouseOver,
}: {
  readonly model: LocalModel
  readonly memoryBytes: number | undefined
  readonly recommendation: Option.Option<LocalModelRecommendation>
  readonly focused: boolean
  readonly pendingDelete: boolean
  readonly installationStarting: boolean
  readonly index: number
  readonly layout: CatalogLayout
  readonly rowId: string
  readonly onClick: () => void
  readonly onMouseOver: () => void
}) {
  const theme = useTheme()
  const status = pendingDelete
    ? "Delete [y/n]"
    : catalogStatus(model, installationStarting)
  const statusColor = pendingDelete
    ? theme.warning
    : model.acquisitionState._tag === "Failed"
      ? theme.error
      : installationStarting
        || model.acquisitionState._tag === "Downloading"
        || model.acquisitionState._tag === "Installed"
        ? theme.primary
        : theme.muted
  const recommendationText = recommendationLabel(recommendation)
  const memoryText = memoryBytes === undefined ? "—" : formatBytes(memoryBytes)
  const speedText = performanceRangeSpeedLabel(model, "t/s")
  const backgroundColor = focused
    ? theme.surfaceHover
    : index % 2 === 0 ? theme.menuBg : theme.menuAltBg

  if (layout.stackedRows) {
    const cursorWidth = 2
    const primaryStatusWidth = layout.mode === "minimal"
      ? 0
      : Math.min(getDisplayWidth(status), Math.max(1, layout.contentWidth - cursorWidth - 1))
    const secondaryStatusWidth = layout.mode === "minimal"
      ? Math.min(getDisplayWidth(`${status} · `), Math.max(1, layout.contentWidth - cursorWidth - 1))
      : 0
    const modelWidth = Math.max(1, layout.contentWidth - cursorWidth - primaryStatusWidth)
    const modelLabel = formatCatalogModelLabel(
      model.presentation.displayName,
      model.presentation.quantizationName,
      modelWidth,
    )
    const metadata = [recommendationText, memoryText, ...(layout.showSpeed ? [speedText] : [])]
      .filter((value) => value !== "")
      .join(" · ")
    const metadataWidth = Math.max(1, layout.contentWidth - cursorWidth - secondaryStatusWidth)

    return (
      <Button
        id={rowId}
        onClick={onClick}
        onMouseOver={onMouseOver}
        style={{
          flexDirection: "column",
          width: "100%",
          height: 2,
          minHeight: 2,
          flexShrink: 0,
          backgroundColor,
        }}
      >
        <box style={{ flexDirection: "row", width: "100%", height: 1, flexShrink: 0 }}>
          <text style={{ fg: focused ? theme.primary : theme.foreground, width: cursorWidth }} wrapMode="none">
            {focused ? "›" : " "}
          </text>
          <text style={{ fg: focused ? theme.primary : theme.foreground, width: modelWidth }} wrapMode="none">
            {modelLabel}
          </text>
          {layout.mode !== "minimal" && (
            <text style={{ fg: statusColor, width: primaryStatusWidth }} wrapMode="none">
              {truncateToDisplayWidth(status, primaryStatusWidth)}
            </text>
          )}
        </box>
        <box style={{ flexDirection: "row", width: "100%", height: 1, flexShrink: 0 }}>
          <text style={{ width: cursorWidth }}> </text>
          {layout.mode === "minimal" && (
            <text style={{ fg: statusColor, width: secondaryStatusWidth }} wrapMode="none">
              {truncateToDisplayWidth(`${status} · `, secondaryStatusWidth)}
            </text>
          )}
          <text style={{ fg: theme.muted, width: metadataWidth }} wrapMode="none">
            {truncateToDisplayWidth(metadata, metadataWidth)}
          </text>
        </box>
      </Button>
    )
  }

  return (
    <Button
      id={rowId}
      onClick={onClick}
      onMouseOver={onMouseOver}
      style={{ flexDirection: "row", width: "100%", backgroundColor }}
    >
      <text style={{ fg: focused ? theme.primary : theme.foreground, width: 2 }} wrapMode="none">
        {focused ? "›" : " "}
      </text>
      <text style={{ fg: focused ? theme.primary : theme.foreground, width: layout.modelWidth }} wrapMode="none">
        {formatCatalogModelLabel(model.presentation.displayName, model.presentation.quantizationName, layout.modelWidth)}
      </text>
      <text style={{ fg: theme.primary, width: layout.columns.recommendation }} wrapMode="none">
        {truncateToDisplayWidth(recommendationText, layout.columns.recommendation)}
      </text>
      <text style={{ fg: theme.muted, width: layout.columns.memory }} wrapMode="none">
        {truncateToDisplayWidth(memoryText, layout.columns.memory)}
      </text>
      {layout.showIntelligence && (
        <text style={{ fg: theme.muted, width: layout.columns.intelligence }} wrapMode="none">
          {truncateToDisplayWidth(intelligenceLabel(model), layout.columns.intelligence)}
        </text>
      )}
      {layout.showQuality && (
        <text style={{ fg: theme.muted, width: layout.columns.quality }} wrapMode="none">
          {truncateToDisplayWidth(qualityLabel(model), layout.columns.quality)}
        </text>
      )}
      {layout.showSpeed && (
        <text style={{ fg: theme.muted, width: layout.columns.speed }} wrapMode="none">
          {truncateToDisplayWidth(speedText, layout.columns.speed)}
        </text>
      )}
      <text style={{ fg: statusColor, width: layout.columns.status }} wrapMode="none">
        {truncateToDisplayWidth(status, layout.columns.status)}
      </text>
    </Button>
  )
})

const CatalogMenu = memo(function CatalogMenu({
  initialCatalogDetailId,
  setRootSwitchingEnabled,
}: CatalogMenuProps) {
  const theme = useTheme()
  const platform = usePlatform()
  const menuSize = useLocalWidth()
  const catalogScrollRef = useRef<ScrollBoxRenderable | null>(null)
  const menuWidth = menuSize.width ?? 80
  const layout = deriveCatalogLayout(menuWidth)
  const snapshot = useLocalModelsSelector(
    selectModelMenusLocalModelsState,
    modelMenusLocalModelsStateEquivalent,
  )
  const modelActions = useLocalModelActions()
  const slotActions = useModelSlotActions()
  const catalogModels = Option.match(snapshot, {
    onNone: () => [] as readonly LocalModel[],
    onSome: ({ models }) => catalogLocalModels(models),
  })
  const recommendationsReady = Option.exists(
    snapshot,
    (models) => models.discoveryState._tag === "Ready",
  )
  const recommendationFor = useCallback((model: LocalModel) =>
    model.servingState._tag === "Assessed"
      ? Option.fromNullable(model.servingState.recommendations[0])
      : Option.none<LocalModelRecommendation>(), [])
  const memoryBytesFor = (model: LocalModel): number | undefined =>
    model.servingState._tag === "Assessed"
      && model.servingState.assessment._tag === "Fits"
      ? model.servingState.assessment.memory.totalRequiredBytes
      : undefined
  const candidates = [...catalogModels].sort((left, right) => {
    const leftInstalled = left.acquisitionState._tag === "Installed"
    const rightInstalled = right.acquisitionState._tag === "Installed"
    return leftInstalled === rightInstalled ? 0 : leftInstalled ? -1 : 1
  })
  const [cursorId, setCursorId] = useState<string | null>(null)
  const [detailId, setDetailId] = useState<string | null>(initialCatalogDetailId)
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null)
  const [hoveredRepositoryUrl, setHoveredRepositoryUrl] = useState<string | null>(null)
  const configurationIdFor = (model: LocalModel) => Option.getOrUndefined(
    localModelConfigurationId(model),
  )
  const cursorIndex = Math.max(0, candidates.findIndex((model) =>
    configurationIdFor(model) === cursorId))
  const cursor = candidates[cursorIndex]
  const detail = candidates.find((model) => configurationIdFor(model) === detailId) ?? null
  const detailConfigurationId = detail === null ? undefined : configurationIdFor(detail)
  const detailMemoryBytes = detail === null ? undefined : memoryBytesFor(detail)
  const installationStartingFor = (model: LocalModel): boolean => {
    const configurationId = configurationIdFor(model)
    return configurationId !== undefined && localModelInstallationIsPending(
      modelActions.installationMutationStates,
      configurationId,
    )
  }
  const detailInstallationStarting = detail !== null
    && installationStartingFor(detail)
  const detailInstallationMutationState = detailConfigurationId === undefined
    ? undefined
    : latestLocalModelInstallationMutationState(
      modelActions.installationMutationStates,
      detailConfigurationId,
    )
  const detailInstallationFailed = detailInstallationMutationState !== undefined
    && Result.isFailure(detailInstallationMutationState.result)
  const progress = Option.match(snapshot, {
    onNone: () => [],
    onSome: (models) => localInferenceProgressLines(models.discoveryState.progress),
  })
  const runningProgress = progress.find((line) => line.state === "running")
  const spinner = useSpinnerFrame(runningProgress !== undefined)
  const detailActions = useMemo(() => {
    if (!detail) return [] as readonly ("primary" | "cancel" | "select")[]
    const actions: ("primary" | "cancel" | "select")[] = []
    if (detail.acquisitionState._tag === "Downloading") actions.push("cancel")
    else if (detailInstallationStarting) return actions
    else if (detail.acquisitionState._tag === "Installed") {
      if (detail.servingState._tag === "Assessed"
        && (detail.servingState.availabilityState._tag === "Selectable"
          || detail.servingState.availabilityState._tag === "Installable")) actions.push("select")
    }
    else actions.push("primary")
    return actions
  }, [detail, detailInstallationStarting])
  const detailActionCursor = useBoundedCursor(detailActions.length)
  const focusedDetailAction = detailActions[detailActionCursor.index]

  const moveCursorTo = useCallback((index: number) => {
    const model = candidates[index]
    const configurationId = model && configurationIdFor(model)
    if (!model || configurationId === undefined) return
    setCursorId(configurationId)
    scrollCatalogCandidateIntoView(catalogScrollRef.current, configurationId)
  }, [candidates])

  const primaryAction = useCallback((model: LocalModel) => {
    const configurationId = configurationIdFor(model)
    if (configurationId === undefined
      || model.acquisitionState._tag === "Downloading"
      || model.acquisitionState._tag === "Installed"
      || localModelInstallationIsPending(
        modelActions.installationMutationStates,
        configurationId,
      )) return
    modelActions.install(configurationId)
  }, [modelActions])

  const selectCandidate = useCallback((model: LocalModel) => {
    if (model.servingState._tag !== "Assessed"
      || model.servingState.assessment._tag !== "Fits") return
    const configurationId = model.servingState.configuration.id
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
      return
    }
    modelActions.installAndAssign(
      configurationId,
      PRIMARY_SLOT_ID,
      Option.getOrElse(reasoningEffort, () => ReasoningEffortSchema.make("none")),
    )
  }, [modelActions, slotActions])

  const runDetailAction = useCallback((action: typeof detailActions[number]) => {
    if (!detail) return
    if (action === "primary") {
      primaryAction(detail)
      return
    }
    if (action === "cancel") {
      if (detail.acquisitionState._tag === "Downloading") {
        modelActions.cancel(detail.acquisitionState.attemptIds)
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
      if (key.name === "escape" || key.name === "backspace" || key.name === "y" || key.name === "n") {
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
      detailActionCursor.reset()
      setDetailId(configurationIdFor(cursor) ?? null)
      setRootSwitchingEnabled(false)
    } else if (key.name === "d" && cursor) {
      key.preventDefault()
      primaryAction(cursor)
    } else if (key.name === "s" && cursor && cursor.servingState._tag === "Assessed"
      && (cursor.servingState.availabilityState._tag === "Selectable"
        || cursor.servingState.availabilityState._tag === "Installable")) {
      key.preventDefault()
      selectCandidate(cursor)
    } else if (key.name === "backspace" && cursor) {
      if (cursor.acquisitionState._tag === "Downloading") {
        modelActions.cancel(cursor.acquisitionState.attemptIds)
        key.preventDefault()
      } else if (cursor.acquisitionState._tag === "Installed") {
        setPendingDeleteId(configurationIdFor(cursor) ?? null)
        key.preventDefault()
      }
    }
  }, [candidates, cursor, cursorIndex, detail, detailActionCursor, detailActions.length, focusedDetailAction, modelActions, moveCursorTo, pendingDeleteId, primaryAction, runDetailAction, selectCandidate, setRootSwitchingEnabled]))

  if (detail) {
    const recommendation = recommendationFor(detail)
    const downloading = detail.acquisitionState._tag === "Downloading"
    const downloaded = detail.acquisitionState._tag === "Installed"
    const failed = detail.acquisitionState._tag === "Failed"
    const detailActionLabel = {
      primary: failed ? "Retry download" : "Download",
      cancel: "Cancel download",
      select: "Select this model",
    } as const
    return (
      <box
        ref={menuSize.ref}
        onSizeChange={menuSize.onSizeChange}
        style={{ flexGrow: 1, minHeight: 0, flexDirection: "column" }}
      >
        <MenuHeader
          title="Catalog"
          selection={detail.presentation.displayName}
          onSectionClick={() => {
            setDetailId(null)
            setRootSwitchingEnabled(true)
          }}
          hints={catalogDetailHints(layout.compactHeader)}
          compact={layout.compactHeader}
          width={menuWidth}
        />
        <scrollbox scrollX={false} style={{
          flexGrow: 1,
          minHeight: 0,
          rootOptions: { backgroundColor: theme.menuBg },
          wrapperOptions: { border: false, backgroundColor: theme.menuBg },
          viewportOptions: { backgroundColor: theme.menuBg },
          contentOptions: { flexDirection: "column", paddingLeft: 2, paddingRight: 2, paddingTop: 1 },
        }}>
          <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD} wrapMode="word">{detail.presentation.displayName}</text>
          <text style={{ fg: theme.muted }} wrapMode="word">{detail.presentation.description}</text>
          {Option.isSome(recommendation) && (
            <>
              <text style={{ fg: theme.primary }}>{recommendationLabel(recommendation)}</text>
              <text style={{ fg: theme.muted }} wrapMode="word">{recommendation.value.explanation}</text>
            </>
          )}
          <text style={{ fg: theme.foreground, marginTop: 1 }} attributes={TextAttributes.BOLD}>Calibrated for this machine</text>
          {layout.compactHeader ? (
            <>
              <text style={{ fg: theme.muted }} wrapMode="word">Memory: {detailMemoryBytes === undefined ? "—" : formatBytes(detailMemoryBytes)}</text>
              <text style={{ fg: theme.muted }} wrapMode="word">Quantization: {detail.presentation.quantization}</text>
              <text style={{ fg: theme.muted }} wrapMode="word">Catalog: {catalogDataLabel(detail)}</text>
              <text style={{ fg: theme.muted }} wrapMode="word">Speed: {performanceRangeSpeedLabel(detail, "tokens/sec")}</text>
            </>
          ) : (
            <>
              <text style={{ fg: theme.muted }} wrapMode="word">
                {detailMemoryBytes === undefined ? "—" : formatBytes(detailMemoryBytes)} memory · {detail.presentation.quantization} · {catalogDataLabel(detail)}
              </text>
              <text style={{ fg: theme.muted }} wrapMode="word">
                {performanceRangeSpeedLabel(detail, "tokens/sec")}
              </text>
            </>
          )}
          <text style={{ fg: failed ? theme.error : detailInstallationStarting || downloading || downloaded ? theme.primary : theme.muted }}>
            {catalogStatus(detail, detailInstallationStarting)}
          </text>
          {failed && <text style={{ fg: theme.error }}>{detail.acquisitionState.failure.message}</text>}
          {detailInstallationFailed && (
            <text style={{ fg: theme.error }}>Failed to install the local model.</text>
          )}
          {Result.isFailure(slotActions.assignResult) && (
            <text style={{ fg: theme.error }}>Failed to update model selection.</text>
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
          <text style={{ fg: theme.muted, marginTop: 1 }} wrapMode="word">License: {Option.getOrElse(detail.presentation.license, () => "Unknown")}</text>
          {huggingFaceRepositoryUrls(detail).map((url) => (
            <box key={url} style={{ flexDirection: "row", alignSelf: "flex-start" }}>
              <text style={{ fg: theme.muted }}>Hugging Face: </text>
              <Button
                onClick={() => { void platform.openLink(url) }}
                onMouseOver={() => setHoveredRepositoryUrl(url)}
                onMouseOut={() => setHoveredRepositoryUrl((hovered) => hovered === url ? null : hovered)}
              >
                <text
                  style={{ fg: hoveredRepositoryUrl === url ? theme.primary : theme.link }}
                  attributes={TextAttributes.UNDERLINE}
                  wrapMode="word"
                >
                  {url}
                </text>
              </Button>
            </box>
          ))}
        </scrollbox>
      </box>
    )
  }

  return (
    <box
      ref={menuSize.ref}
      onSizeChange={menuSize.onSizeChange}
      style={{ flexGrow: 1, minHeight: 0, flexDirection: "column" }}
    >
      <MenuHeader
        title="Catalog"
        subtitle={layout.stackedRows
          ? undefined
          : layout.mode === "full" ? "Find and download local models" : "Local models"}
        summary={layout.mode === "minimal"
          ? String(candidates.length)
          : layout.compactHeader ? `${candidates.length} models` : `${candidates.length} compatible`}
        hints={catalogListHints(layout.mode)}
        compact={layout.compactHeader}
        width={menuWidth}
      />
      <scrollbox ref={catalogScrollRef} scrollX={false} style={{
        flexGrow: 1,
        minHeight: 0,
        rootOptions: { backgroundColor: theme.menuBg },
        wrapperOptions: { border: false, backgroundColor: theme.menuBg },
        viewportOptions: { backgroundColor: theme.menuBg },
        contentOptions: { flexDirection: "column", paddingLeft: 2, paddingRight: 2, paddingTop: 1 },
      }}>
        {!layout.stackedRows && (
          <box style={{ flexDirection: "row", width: "100%" }}>
            <text style={{ fg: theme.muted, width: 2 }} wrapMode="none"> </text>
            <text style={{ fg: theme.muted, width: layout.modelWidth }} wrapMode="none">MODEL</text>
            <text style={{ fg: theme.muted, width: layout.columns.recommendation }} wrapMode="none">RECOMMENDATION</text>
            <text style={{ fg: theme.muted, width: layout.columns.memory }} wrapMode="none">MEMORY</text>
            {layout.showIntelligence && (
              <text style={{ fg: theme.muted, width: layout.columns.intelligence }} wrapMode="none">INTELLIGENCE</text>
            )}
            {layout.showQuality && (
              <text style={{ fg: theme.muted, width: layout.columns.quality }} wrapMode="none">QUALITY</text>
            )}
            {layout.showSpeed && (
              <text style={{ fg: theme.muted, width: layout.columns.speed }} wrapMode="none">SPEED</text>
            )}
            <text style={{ fg: theme.muted, width: layout.columns.status }} wrapMode="none">STATUS</text>
          </box>
        )}
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
          const configurationId = configurationIdFor(candidate)
          if (configurationId === undefined) return null
          const focused = index === cursorIndex
          const pendingDelete = pendingDeleteId === configurationId
          return (
            <CatalogCandidateRow
              key={configurationId}
              model={candidate}
              memoryBytes={memoryBytesFor(candidate)}
              recommendation={recommendationFor(candidate)}
              focused={focused}
              pendingDelete={pendingDelete}
              installationStarting={installationStartingFor(candidate)}
              index={index}
              layout={layout}
              rowId={catalogCandidateRowId(configurationId)}
              onClick={() => {
                setPendingDeleteId(null)
                detailActionCursor.reset()
                setDetailId(configurationId)
                setRootSwitchingEnabled(false)
              }}
              onMouseOver={() => {
                setCursorId(configurationId)
                if (pendingDeleteId !== configurationId) setPendingDeleteId(null)
              }}
            />
          )
        })}
      </scrollbox>
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
                    ? <text style={{ fg: theme.muted }}>{currentSlot.value.availability.failure.message}</text>
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
