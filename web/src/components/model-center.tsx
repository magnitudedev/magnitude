import { useMemo, useState, type ReactNode } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Progress } from "@/components/ui/progress"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Option } from "effect"
import { Result } from "@effect-atom/atom-react"
import {
  AlertTriangle,
  Cpu,
  Download,
  EllipsisVertical,
  FolderOpen,
  Layers3,
  Loader2,
  MemoryStick,
  PackageOpen,
  RefreshCw,
  Search,
  Trash2,
  X,
} from "lucide-react"
import {
  deriveHardwareMemoryView,
  formatLocalModelDisplayName,
  localModelRadarAxes,
  localModelIsInstalled,
  localModelServingState,
  localModelStorageBytes,
  modelDownloadFailureMessage,
  modelSlotResidentAllocation,
  useCatalogModels,
  useLocalInferenceHardware,
  useLocalModelActions,
  useLocalModels,
  useModelSlots,
  usePlatform,
  usePreviewModelLoad,
} from "@magnitudedev/client-common"
import {
  PRIMARY_SLOT_ID,
  acquisitionFailure,
  acquisitionProgress,
  installedAcquisition,
  type LocalModel,
} from "@magnitudedev/sdk"
import type { SettingsTab } from "../state/web-atoms"
import {
  formatBytes,
  formatContext,
  modelContextLength,
  slotStatus,
  transferProgress,
} from "./local-inference-format"
import { ModelRadarChart } from "./model-radar-chart"
const valueOf = <A, E>(result: Result.Result<A, E>): A | null =>
  Option.getOrNull(Result.value(result))
const modelKey = (model: LocalModel): string => model.modelId
type CatalogLocalModel = Extract<LocalModel, { readonly _tag: "Catalog" }>
const servingState = (model: LocalModel) => Option.getOrUndefined(localModelServingState(model))

function QueryNotice({
  result,
  label,
}: {
  readonly result: Result.Result<unknown, unknown>
  readonly label: string
}): ReactNode {
  if (Result.isFailure(result)) {
    return (
      <div className="flex items-center gap-2 rounded-[7px] border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-850 px-3 py-2.5 text-slate-600 dark:text-slate-400 text-xs [&.danger]:border-red-300 [&.danger]:text-red-600 dark:[&.danger]:border-red-700 dark:[&.danger]:text-red-400 danger">
        <AlertTriangle size={15} />
        Unable to load {label}.
      </div>
    )
  }
  if (Option.isNone(Result.value(result))) {
    return (
      <div className="flex items-center gap-2 rounded-[7px] border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-850 px-3 py-2.5 text-slate-600 dark:text-slate-400 text-xs [&.danger]:border-red-300 [&.danger]:text-red-600 dark:[&.danger]:border-red-700 dark:[&.danger]:text-red-400">
        <Loader2 className="animate-spin" size={15} />
        Loading {label}…
      </div>
    )
  }
  return null
}

function LoadingNotice({
  title,
  description,
}: {
  readonly title: string
  readonly description: string
}): ReactNode {
  return (
    <section
      className="flex items-center gap-3.5 rounded-lg border border-slate-300 bg-white px-[18px] py-4 dark:border-slate-750 dark:bg-slate-850 [&_h2]:mb-1 [&_h2]:text-[19px] [&_h2]:text-slate-900 dark:[&_h2]:text-slate-200 [&_p]:text-xs [&_p]:text-slate-600 dark:[&_p]:text-slate-400"
      role="status"
      aria-live="polite"
    >
      <Loader2 className="shrink-0 animate-spin" size={20} aria-hidden="true" />
      <div>
        <h2>{title}</h2>
        <p>{description}</p>
      </div>
    </section>
  )
}
const modelFailure = (model: LocalModel): string | null => {
  if (model._tag === "Catalog" && model.acquisitionState._tag === "RemoveFailed")
    return model.acquisitionState.failure.message
  const transferFailure = model._tag === "Catalog"
    ? acquisitionFailure(model.acquisitionState)
    : undefined
  if (transferFailure !== undefined)
    return modelDownloadFailureMessage(transferFailure)
  if (model._tag === "Discovered" && model.state._tag !== "Ready") return model.state.failure.message
  const serving = servingState(model)
  if (serving?._tag === "Failed") return serving.failure.message
  if (serving?._tag === "Assessed") {
    if (serving.assessment._tag === "DoesNotFit") {
      return `Needs ${formatBytes(
        serving.assessment.deficitBytes
      )} more ${serving.assessment.limitingResource}.`
    }
    if (serving.assessment._tag === "Incompatible")
      return serving.assessment.failure.message
  }
  return null
}
const modelTransfer = (model: LocalModel) =>
  model._tag === "Catalog" ? acquisitionProgress(model.acquisitionState) ?? null : null
type StatusTone = "neutral" | "success" | "progress" | "warning" | "danger"
const modelStatus = (model: LocalModel): { readonly label: string; readonly tone: StatusTone } => {
  if (model._tag === "Discovered" && model.state._tag === "Ambiguous")
    return { label: "Ambiguous", tone: "danger" }
  if (model._tag === "Discovered" && model.state._tag === "Unavailable")
    return { label: "Unavailable", tone: "danger" }
  const acquisition = model._tag === "Catalog" ? model.acquisitionState : undefined
  if (acquisition?._tag === "Removing")
    return {
      label: "Removing",
      tone: "progress",
    }
  if (acquisition?._tag === "RemoveFailed")
    return {
      label: "Remove failed",
      tone: "danger",
    }
  const serving = servingState(model)
  if (serving?._tag === "Assessing")
    return {
      label: "Assessing",
      tone: "progress",
    }
  if (serving?._tag === "Failed")
    return {
      label: "Assessment failed",
      tone: "danger",
    }
  if (serving?._tag === "Assessed" && serving.assessment._tag === "DoesNotFit")
    return {
      label: "Doesn’t fit",
      tone: "danger",
    }
  if (serving?._tag === "Assessed" && serving.assessment._tag === "Incompatible")
    return {
      label: "Incompatible",
      tone: "danger",
    }
  if (acquisition?._tag === "UpdateAvailable")
    return {
      label: "Update available",
      tone: "warning",
    }
  if ((model._tag === "Discovered" && model.state._tag === "Ready") ||
    (acquisition !== undefined && installedAcquisition(acquisition) !== undefined
      && acquisition._tag !== "Updating" && acquisition._tag !== "UpdateFailed"))
    return {
      label: "Installed",
      tone: "success",
    }
  if (acquisition?._tag === "Installing")
    return {
      label: `Downloading ${transferProgress(acquisition.progress)}%`,
      tone: "progress",
    }
  if (acquisition?._tag === "Updating")
    return {
      label: `Updating ${transferProgress(acquisition.progress)}%`,
      tone: "progress",
    }
  if (acquisition?._tag === "InstallFailed")
    return {
      label: "Download failed",
      tone: "danger",
    }
  if (acquisition?._tag === "UpdateFailed")
    return {
      label: "Update failed",
      tone: "danger",
    }
  return {
    label: "Available",
    tone: "neutral",
  }
}
const statusToneClass = (tone: string): string => {
  switch (tone) {
    case "success":
      return "text-green-700 dark:text-green-500"
    case "progress":
      return "text-blue-700 dark:text-blue-500"
    case "warning":
      return "text-orange-700 dark:text-orange-500"
    case "danger":
      return "text-red-600 dark:text-red-500"
    case "neutral":
    default:
      return "text-slate-600 dark:text-slate-400"
  }
}
type ModelTransferProgressValue = {
  readonly completedBytes: number
  readonly totalBytes: number
  readonly stage?: string
}
function ModelTransferProgress({
  transfer,
  ariaLabel,
  showStage = true,
}: {
  readonly transfer: ModelTransferProgressValue
  readonly ariaLabel: string
  readonly showStage?: boolean
}): ReactNode {
  const progress = transferProgress(transfer)
  return (
    <div className="grid w-full gap-2.5">
      <div className="flex min-w-0 items-center justify-between gap-4 font-sans text-[12px] leading-4">
        <div className="flex min-w-0 items-baseline gap-1.5 text-slate-500">
          {showStage ? (
            <>
              <span className="shrink-0 font-medium capitalize text-slate-700 dark:text-slate-300">
                {transfer.stage ?? "Downloading"}
              </span>
              <span aria-hidden="true">·</span>
            </>
          ) : null}
          <span className="truncate">
            {formatBytes(transfer.completedBytes)} of {formatBytes(transfer.totalBytes)}
          </span>
        </div>
        <strong className="shrink-0 font-semibold tabular-nums text-blue-700 dark:text-blue-400">
          {progress}%
        </strong>
      </div>
      <Progress
        value={progress}
        max={100}
        aria-label={ariaLabel}
        className="block w-full"
        trackClassName="h-2 rounded-full bg-slate-250 dark:bg-slate-700"
        indicatorClassName="rounded-full bg-blue-600 dark:bg-blue-500"
      />
    </div>
  )
}
const installedModelTargetPath = (model: LocalModel): string | null => {
  if (model._tag === "Discovered") return model.state._tag !== "Ambiguous"
    ? model.state.installation.primaryPath
    : null
  const installation = installedAcquisition(model.acquisitionState)?.installation
  return installation?._tag === "Resolved" ? installation.primaryPath : null
}

function InstalledModelMenu({
  model,
}: {
  readonly model: LocalModel
}): ReactNode {
  const platform = usePlatform()
  const actions = useLocalModelActions()
  const [confirmingRemoval, setConfirmingRemoval] = useState(false)
  const configurationId = model.modelId
  const displayName = formatLocalModelDisplayName(model)
  const installedPath = installedModelTargetPath(model)
  const ownership = model._tag === "Catalog"
    ? installedAcquisition(model.acquisitionState)?.installation.ownership
    : undefined
  const externallyManaged = model._tag === "Discovered" || ownership === "ExternalHuggingFace"
    || ownership === "Mixed"

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger
          render={
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label={`Actions for ${displayName}`}
            />
          }
        >
          <EllipsisVertical aria-hidden="true" />
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-44">
          <DropdownMenuItem
            disabled={installedPath === null || platform.id === "web"}
            onClick={() => {
              if (installedPath !== null) platform.showItemInFolder(installedPath)
            }}
          >
            <FolderOpen aria-hidden="true" />
            Reveal in Finder
          </DropdownMenuItem>
          {!externallyManaged ? (
            <>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                variant="destructive"
                disabled={configurationId === null}
                onClick={() => setConfirmingRemoval(true)}
              >
                <Trash2 aria-hidden="true" />
                Remove Model
              </DropdownMenuItem>
            </>
          ) : null}
        </DropdownMenuContent>
      </DropdownMenu>

      {!externallyManaged ? <AlertDialog open={confirmingRemoval} onOpenChange={setConfirmingRemoval}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Remove {displayName}?</AlertDialogTitle>
            <AlertDialogDescription>
              This deletes the downloaded model files from this computer.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              type="button"
              variant="destructive"
              onClick={() => {
                if (model._tag !== "Catalog") return
                actions.remove(model.modelId)
                setConfirmingRemoval(false)
              }}
            >
              Remove Model
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog> : null}
    </>
  )
}

function InstalledLibrary({
  models,
}: {
  readonly models: readonly LocalModel[]
}): ReactNode {
  const [query, setQuery] = useState("")
  const normalizedQuery = query.trim().toLowerCase()
  const filteredModels = normalizedQuery
    ? models.filter((model) =>
        formatLocalModelDisplayName(model)
          .toLowerCase()
          .includes(normalizedQuery)
      )
    : models

  return (
    <section
      className="flex flex-col gap-5"
      aria-labelledby="installed-models-title"
    >
      <div className="flex items-center justify-between gap-5 max-[640px]:items-start max-[640px]:flex-col">
        <h2
          id="installed-models-title"
          className="font-heading text-[22px] leading-tight tracking-[-.02em] text-slate-900 dark:text-slate-100"
        >
          Installed models
        </h2>
        <div className="flex w-full max-w-[390px] items-center justify-end gap-3">
          <span className="shrink-0 font-sans text-xs tabular-nums text-slate-500">
            {filteredModels.length} {filteredModels.length === 1 ? "model" : "models"}
          </span>
          <div className="relative w-full max-w-[300px]">
            <Search
              size={15}
              aria-hidden="true"
              className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-500"
            />
            <Input
              type="search"
              value={query}
              onChange={(event) => setQuery(event.currentTarget.value)}
              placeholder="Search installed models"
              aria-label="Search installed models"
              className="pl-8"
            />
          </div>
        </div>
      </div>
      {models.length === 0 ? (
        <div className="rounded-[10px] border border-dashed border-slate-300 dark:border-slate-750 bg-white dark:bg-slate-850 p-[26px] text-center text-[13px] text-slate-500">
          No models are installed yet. Open Catalog to choose one for this
          machine.
        </div>
      ) : filteredModels.length === 0 ? (
        <div className="rounded-[10px] border border-dashed border-slate-300 dark:border-slate-750 p-[26px] text-center text-[13px] text-slate-500">
          No installed models match “{query.trim()}”.
        </div>
      ) : (
        <div
          className="overflow-hidden rounded-lg border border-slate-300 bg-white dark:border-slate-750 dark:bg-slate-850"
          role="list"
        >
          {filteredModels.map((model) => {
            const displayName = formatLocalModelDisplayName(model)
            const context = modelContextLength(model)
            const installedBytes = Option.getOrNull(localModelStorageBytes(model))
            return (
              <article
                className="grid min-h-[82px] grid-cols-[auto_minmax(240px,1fr)_minmax(190px,.45fr)_auto] items-center gap-4 border-b border-slate-200 px-4 py-3.5 text-slate-500 last:border-b-0 dark:border-slate-800 max-[820px]:grid-cols-[auto_minmax(0,1fr)_auto]"
                role="listitem"
                key={modelKey(model)}
              >
                <PackageOpen size={17} aria-hidden="true" />
                <div className="flex min-w-0 flex-col gap-[3px]">
                  <strong className="break-words text-[13px] font-semibold text-slate-900 dark:text-slate-200">
                    {displayName}
                  </strong>
                  <span className="overflow-hidden text-ellipsis whitespace-nowrap text-[11px] text-slate-500">
                    {model.presentation.description}
                  </span>
                </div>
                <dl className="grid grid-cols-2 gap-5 max-[820px]:col-start-2">
                  <div className="min-w-0">
                    <dt className="text-[9px] uppercase tracking-[.06em] text-slate-500">
                      Context
                    </dt>
                    <dd className="whitespace-nowrap text-[11px] text-slate-600 dark:text-slate-400">
                      {context === null ? "Pending" : formatContext(context)}
                    </dd>
                  </div>
                  <div className="min-w-0">
                    <dt className="text-[9px] uppercase tracking-[.06em] text-slate-500">
                      Storage
                    </dt>
                    <dd className="whitespace-nowrap text-[11px] tabular-nums text-slate-600 dark:text-slate-400">
                      {installedBytes === null ? "Unknown" : formatBytes(installedBytes)}
                    </dd>
                  </div>
                </dl>
                <InstalledModelMenu model={model} />
              </article>
            )
          })}
        </div>
      )}
    </section>
  )
}
function ModelsView(): ReactNode {
  const modelsResult = useLocalModels()
  const models = valueOf(modelsResult)
  const inventoryLoading = models !== null && !models.reconciliationComplete
  const installed =
    models?.models.filter(
      localModelIsInstalled
    ) ?? []
  return (
    <div className="box-border mx-auto flex w-full max-w-[1240px] flex-col gap-[34px] px-[clamp(18px,4vw,48px)] pt-[34px] pb-[72px] max-[640px]:pt-16 max-[620px]:px-3.5">
      {models === null && !Result.isFailure(modelsResult) ? (
        <LoadingNotice
          title="Loading models"
          description="Checking the models available on this computer."
        />
      ) : (
        <QueryNotice result={modelsResult} label="models" />
      )}
      {models && inventoryLoading ? (
        <LoadingNotice
          title="Loading models"
          description="Reading the models installed on this computer."
        />
      ) : null}
      {models && !inventoryLoading ? (
        <InstalledLibrary models={installed} />
      ) : null}
    </div>
  )
}
const repositoryUrl = (model: LocalModel): string | null =>
  model.presentation.sourceUrls.find((url) => url.startsWith("https://huggingface.co/")) ?? null
type CatalogFilter = "all" | "installed"
type CatalogSort =
  | "recent"
  | "intelligence"
  | "largest"
  | "smallest"
  | "name"

const catalogSortLabels: Readonly<Record<CatalogSort, string>> = {
  recent: "Newest",
  intelligence: "Most intelligent",
  largest: "Largest download",
  smallest: "Smallest download",
  name: "Name",
}

const catalogData = (model: LocalModel) =>
  model._tag === "Catalog" ? model.catalogData : null

const isCatalogVisible = (model: LocalModel): boolean =>
  Option.match(localModelServingState(model), {
    onNone: () => true,
    onSome: (serving) => serving._tag !== "Assessed" || serving.assessment._tag === "Fits",
  })

const matchesCatalogFilter = (
  model: CatalogLocalModel,
  filter: CatalogFilter
): boolean => {
  if (filter === "all") return true
  return installedAcquisition(model.acquisitionState) !== undefined
}

const compareCatalogModels = (
  left: CatalogLocalModel,
  right: CatalogLocalModel,
  sort: CatalogSort
): number => {
  const leftName = formatLocalModelDisplayName(left)
  const rightName = formatLocalModelDisplayName(right)
  const byName = leftName.localeCompare(rightName)
  const leftCatalog = catalogData(left)
  const rightCatalog = catalogData(right)
  if (sort === "name") return byName
  if (sort === "recent") {
    return (
      String(rightCatalog?.releaseDate ?? "").localeCompare(
        String(leftCatalog?.releaseDate ?? "")
      ) || byName
    )
  }
  if (sort === "intelligence") {
    return (
      (rightCatalog?.intelligence.score ?? -1) -
        (leftCatalog?.intelligence.score ?? -1) || byName
    )
  }
  if (sort === "largest") {
    return right.storageBytes - left.storageBytes || byName
  }
  if (sort === "smallest") {
    return left.storageBytes - right.storageBytes || byName
  }
  return byName
}

function CatalogCandidate({
  model,
  selected,
  onSelect,
}: {
  readonly model: CatalogLocalModel
  readonly selected: boolean
  readonly onSelect: () => void
}): ReactNode {
  const status = modelStatus(model)
  return (
    <Button
      variant="unstyled"
      size="unstyled"
      type="button"
      className="flex min-h-[76px] w-full items-center border-b border-slate-200 bg-transparent px-5 py-3.5 text-left font-sans text-slate-600 outline-none transition-colors last:border-b-0 hover:bg-slate-100 focus-visible:bg-slate-100 data-[selected=true]:bg-slate-150 dark:border-slate-800 dark:text-slate-400 dark:hover:bg-slate-800 dark:focus-visible:bg-slate-800 dark:data-[selected=true]:bg-slate-750"
      data-selected={selected}
      aria-pressed={selected}
      onClick={onSelect}
    >
      <span className="flex min-w-0 flex-1 flex-col gap-1.5">
        {status.tone !== "neutral" ? (
          <span className="flex items-center justify-between gap-3 text-[10px] font-semibold uppercase leading-none tracking-[.07em]">
            <span />
            <span
              className={`${statusToneClass(
                status.tone
              )} shrink-0 tracking-normal normal-case`}
            >
              {status.label}
            </span>
          </span>
        ) : null}
        <strong className="text-[13px] font-semibold leading-[1.4] text-slate-900 [overflow-wrap:anywhere] dark:text-slate-100">
          {formatLocalModelDisplayName(model)}
        </strong>
      </span>
    </Button>
  )
}
function CatalogInspector({
  model,
}: {
  readonly model: CatalogLocalModel
}): ReactNode {
  const modelActions = useLocalModelActions()
  const configurationId = model.modelId
  const status = modelStatus(model)
  const axes = Option.getOrNull(localModelRadarAxes(model))
  const transfer = modelTransfer(model)
  const installed = installedAcquisition(model.acquisitionState) !== undefined
  const starting = model.acquisitionState._tag === "Removing"
  const source = repositoryUrl(model)
  const actions = (
    <div className="flex shrink-0 items-center gap-2">
      {status.tone !== "neutral" ? (
        <span
          className={`${statusToneClass(
            status.tone
          )} text-[11px] font-medium whitespace-nowrap ${
            transfer ? "capitalize" : ""
          }`}
        >
          {transfer ? transfer.stage ?? "Downloading" : status.label}
        </span>
      ) : null}
      {!installed && !transfer && configurationId && (
        <Button
          variant="default"
          size="default"
          type="button"
          disabled={starting}
          onClick={() => modelActions.install(configurationId)}
        >
          {model.acquisitionState._tag === "InstallFailed" ? (
            <RefreshCw size={14} />
          ) : (
            <Download size={14} />
          )}
          {model.acquisitionState._tag === "InstallFailed"
            ? "Retry download"
            : "Download"}
        </Button>
      )}
      {(model.acquisitionState._tag === "UpdateAvailable" ||
        model.acquisitionState._tag === "UpdateFailed") &&
        configurationId && (
          <Button
            variant="default"
            size="default"
            type="button"
            disabled={starting}
            onClick={() => modelActions.install(configurationId)}
          >
            <RefreshCw size={14} />
            {model.acquisitionState._tag === "UpdateFailed" ? "Retry update" : "Update"}
          </Button>
        )}
      {transfer && configurationId && (
        <Button
          variant="outline"
          size="default"
          type="button"
          onClick={() => modelActions.cancel(configurationId)}
        >
          <X size={14} /> Cancel
        </Button>
      )}
    </div>
  )
  return (
    <article className="grid h-full min-h-0 min-w-0 grid-rows-[auto_minmax(0,1fr)] overflow-hidden bg-white dark:bg-slate-850 max-[840px]:h-auto max-[840px]:overflow-visible">
      <header className="border-b border-slate-300 px-7 py-6 dark:border-slate-750 max-[620px]:px-4">
        <div className="flex items-start justify-between gap-6 max-[620px]:flex-col">
          <div className="min-w-0 max-w-[720px]">
            <h2 className="font-heading text-[24px] leading-[1.2] tracking-[-.025em] text-slate-900 [overflow-wrap:anywhere] dark:text-slate-100">
              {formatLocalModelDisplayName(model)}
            </h2>
            <p className="mt-2 text-[13px] leading-5 text-slate-600 dark:text-slate-400">
              {model.presentation.description}
            </p>
          </div>
          {actions}
        </div>
      </header>

      <div className="min-h-0 overflow-x-hidden overflow-y-auto px-7 py-6 max-[840px]:overflow-visible max-[620px]:p-4">
        {transfer && (
          <div className="mb-6 border-b border-slate-200 pb-5 dark:border-slate-800">
            <ModelTransferProgress
              transfer={transfer}
              showStage={false}
              ariaLabel={`${formatLocalModelDisplayName(model)} transfer progress`}
            />
          </div>
        )}
        {modelFailure(model) && (
          <p className="!text-red-600 dark:!text-red-500">
            {modelFailure(model)}
          </p>
        )}

        {axes ? (
          <section
            className="min-w-0"
            aria-labelledby="model-profile-title"
          >
            <h3
              id="model-profile-title"
              className="font-heading text-[18px] text-slate-900 dark:text-slate-100"
            >
              Model profile
            </h3>
            <ModelRadarChart axes={axes} />
          </section>
        ) : (
          <div className="mt-7 flex items-center gap-2 text-[11px] text-slate-500">
            <AlertTriangle size={15} />A complete comparison profile is not
            available for this configuration.
          </div>
        )}

        <dl className="flex flex-wrap gap-x-10 gap-y-4 border-t border-slate-300 pt-5 dark:border-slate-750 [&_div]:min-w-[120px] [&_dt]:text-[10px] [&_dt]:font-medium [&_dt]:uppercase [&_dt]:tracking-[.06em] [&_dt]:text-slate-500 [&_dd]:mt-1.5 [&_dd]:text-[12px] [&_dd]:text-slate-600 dark:[&_dd]:text-slate-400 [&_a]:text-blue-700 [&_a]:no-underline hover:[&_a]:underline dark:[&_a]:text-blue-400">
          <div>
            <dt>License</dt>
            <dd>
              {Option.getOrElse(model.presentation.license, () => "Unknown")}
            </dd>
          </div>
          {source && (
            <div>
              <dt>Source</dt>
              <dd>
                <a href={source} target="_blank" rel="noreferrer">
                  Hugging Face ↗
                </a>
              </dd>
            </div>
          )}
        </dl>
      </div>
    </article>
  )
}
function CatalogView(): ReactNode {
  const catalogResult = useCatalogModels()
  const modelActions = useLocalModelActions()
  const catalog = valueOf(catalogResult)
  const [selectedKey, setSelectedKey] = useState<string | null>(null)
  const [query, setQuery] = useState("")
  const [filter, setFilter] = useState<CatalogFilter>("all")
  const [sort, setSort] = useState<CatalogSort>("intelligence")
  const normalizedQuery = query.trim().toLowerCase()
  const allCandidates = catalog?.models.filter(
    (model): model is CatalogLocalModel => model._tag === "Catalog"
  ) ?? []
  const visibleCandidates = useMemo(
    () => allCandidates.filter(isCatalogVisible),
    [allCandidates]
  )
  const candidates = useMemo(
    () =>
      visibleCandidates
        .filter(
          (candidate) =>
            matchesCatalogFilter(candidate, filter) &&
            (normalizedQuery.length === 0 ||
              formatLocalModelDisplayName(candidate)
                .toLowerCase()
                .includes(normalizedQuery))
        )
        .toSorted((left, right) => compareCatalogModels(left, right, sort)),
    [filter, normalizedQuery, sort, visibleCandidates]
  )
  const selected =
    candidates.find((model) => modelKey(model) === selectedKey) ??
    candidates[0] ??
    null
  const reconciliationComplete = catalog?.reconciliationComplete ?? false
  const filterCounts = useMemo(
    () => ({
      all: visibleCandidates.length,
      installed: visibleCandidates.filter((candidate) =>
        matchesCatalogFilter(candidate, "installed")
      ).length,
    }),
    [visibleCandidates]
  )
  return (
    <div className="box-border mx-auto flex h-full min-h-0 w-full max-w-[1500px] flex-col gap-5 overflow-hidden px-[clamp(18px,3vw,42px)] py-7 max-[840px]:h-auto max-[840px]:overflow-visible max-[640px]:pt-16 max-[620px]:px-3.5">
      {catalog === null && !Result.isFailure(catalogResult) ? (
        <LoadingNotice
          title="Loading catalog"
          description="Checking the model catalog for this computer."
        />
      ) : (
        <QueryNotice result={catalogResult} label="catalog" />
      )}
      {catalog !== null && !reconciliationComplete ? (
        <LoadingNotice
          title="Loading catalog"
          description="Assessing local models for this computer."
        />
      ) : null}
      {catalog && reconciliationComplete ? (
        <>
          <header className="shrink-0">
            <div>
              <h2 className="font-heading text-[24px] leading-tight tracking-[-.02em] text-slate-900 dark:text-slate-100">
                Catalog
              </h2>
              <p className="mt-1 text-[13px] text-slate-600 dark:text-slate-400">
                Models assessed for this computer.
              </p>
            </div>
            <div className="mt-5 flex min-w-0 flex-wrap items-center gap-3">
              <div className="relative w-[280px] max-[620px]:w-full">
                <Search
                  size={15}
                  aria-hidden="true"
                  className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-500"
                />
                <Input
                  type="search"
                  value={query}
                  onChange={(event) => setQuery(event.currentTarget.value)}
                  placeholder="Search catalog"
                  aria-label="Search catalog"
                  className="pl-8"
                />
              </div>
              <div
                className="flex h-8 items-center rounded-md bg-slate-100 p-0.5 dark:bg-slate-800"
                role="group"
                aria-label="Filter catalog"
              >
                {(
                  [
                    ["all", "All"],
                    ["installed", "Installed"],
                  ] as const
                ).map(([value, label]) => (
                  <Button
                    key={value}
                    variant="unstyled"
                    size="unstyled"
                    type="button"
                    aria-pressed={filter === value}
                    onClick={() => setFilter(value)}
                    className="h-7 rounded-[5px] px-2.5 text-[11px] font-medium text-slate-600 transition-colors hover:text-slate-900 aria-pressed:bg-white aria-pressed:text-slate-900 aria-pressed:shadow-sm dark:text-slate-400 dark:hover:text-slate-100 dark:aria-pressed:bg-slate-750 dark:aria-pressed:text-slate-100"
                  >
                    {label}
                    <span className="ml-1.5 tabular-nums text-slate-500">
                      {filterCounts[value]}
                    </span>
                  </Button>
                ))}
              </div>
              <div className="ml-auto flex items-center gap-2 max-[760px]:ml-0">
                <span className="text-[11px] text-slate-500">Sort by</span>
                <Select
                  value={sort}
                  onValueChange={(value) => setSort(value as CatalogSort)}
                >
                  <SelectTrigger aria-label="Sort catalog" className="w-[168px]">
                    <SelectValue>{catalogSortLabels[sort]}</SelectValue>
                  </SelectTrigger>
                  <SelectContent align="end" className="w-[196px]">
                    <SelectItem value="recent">Newest</SelectItem>
                    <SelectItem value="intelligence">Most intelligent</SelectItem>
                    <SelectItem value="largest">Largest download</SelectItem>
                    <SelectItem value="smallest">Smallest download</SelectItem>
                    <SelectItem value="name">Name</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </header>
          {visibleCandidates.length === 0 ? (
            <div className="rounded-[10px] border border-dashed border-slate-300 dark:border-slate-750 bg-white dark:bg-slate-850 p-[26px] text-center text-[13px] text-slate-500">
              No local catalog models are currently available.
            </div>
          ) : candidates.length === 0 ? (
            <div className="flex min-h-[240px] flex-1 flex-col items-center justify-center rounded-lg border border-dashed border-slate-300 px-6 text-center dark:border-slate-750">
              <p className="text-[13px] text-slate-600 dark:text-slate-400">
                No catalog models match these controls.
              </p>
              <Button
                variant="ghost"
                size="sm"
                className="mt-2"
                onClick={() => {
                  setQuery("")
                  setFilter("all")
                }}
              >
                Clear search and filter
              </Button>
            </div>
          ) : candidates.length > 0 ? (
            <div className="grid min-h-0 flex-1 grid-cols-[minmax(300px,.66fr)_minmax(0,1.34fr)] items-stretch overflow-hidden rounded-lg border border-slate-300 bg-white dark:border-slate-750 dark:bg-slate-850 max-[1050px]:grid-cols-[minmax(280px,.76fr)_minmax(0,1.24fr)] max-[840px]:grid-cols-1 max-[840px]:overflow-visible">
              <section className="min-h-0 min-w-0 border-r border-slate-300 dark:border-slate-750 max-[840px]:border-r-0 max-[840px]:border-b">
                <div
                  className="h-full min-h-0 overflow-y-auto max-[840px]:max-h-[420px]"
                  role="group"
                  aria-label="Catalog models"
                >
                  {candidates.map((candidate) => {
                    const key = modelKey(candidate)
                    return (
                      <CatalogCandidate
                        key={key}
                        model={candidate}
                        selected={
                          selected !== null && modelKey(selected) === key
                        }
                        onSelect={() => setSelectedKey(key)}
                      />
                    )
                  })}
                </div>
              </section>
              {selected && <CatalogInspector model={selected} />}
            </div>
          ) : null}
        </>
      ) : null}
    </div>
  )
}
const percentage = (part: number, total: number): number =>
  total <= 0 ? 0 : Math.min(100, Math.max(0, (part / total) * 100))
function HardwareView(): ReactNode {
  const hardwareResult = useLocalInferenceHardware()
  const slotsResult = useModelSlots()
  const preview = usePreviewModelLoad(PRIMARY_SLOT_ID)
  const hardware = valueOf(hardwareResult)
  const slots = valueOf(slotsResult)
  const primary = slots?.slots.primary ?? null
  const allocation = primary
    ? modelSlotResidentAllocation(primary)
    : Option.none()
  const memory = hardware
    ? deriveHardwareMemoryView(hardware, allocation)
    : null
  return (
    <div className="box-border mx-auto flex w-full max-w-[1240px] flex-col gap-[34px] px-[clamp(18px,4vw,48px)] pt-[34px] pb-[72px] max-[640px]:pt-16 max-[620px]:px-3.5">
      <QueryNotice result={hardwareResult} label="hardware" />
      <QueryNotice result={slotsResult} label="model runtime" />
      {hardware && (
        <>
          <section className="flex items-center gap-4 border-b border-slate-300 dark:border-slate-750 pb-[26px] max-[620px]:flex-wrap max-[620px]:items-start [&_h2]:mb-1 [&_h2]:text-[19px] [&_h2]:leading-tight [&_h2]:tracking-[-.015em] [&_h2]:text-slate-900 dark:[&_h2]:text-slate-200 [&_p]:text-[12px] [&_p]:leading-normal [&_p]:text-slate-600 dark:[&_p]:text-slate-400">
            <Cpu
              className="shrink-0 text-blue-700 dark:text-blue-500"
              size={24}
              aria-hidden="true"
            />
            <div className="min-w-0">
              <span className="block text-slate-500 font-sans text-[10px] font-[650] leading-[1.2] tracking-[.09em] uppercase mb-[5px]">
                This machine
              </span>
              <h2>
                {Option.getOrElse(hardware.productName, () =>
                  Option.getOrElse(hardware.processor, () => "Local system")
                )}
              </h2>
              <p>
                {hardware.platform} · {hardware.architecture} ·{" "}
                {hardware.logicalCores} logical cores
              </p>
            </div>
            <div className="ml-auto text-right max-[620px]:ml-[62px] max-[620px]:w-full max-[620px]:text-left [&_strong]:block [&_strong]:font-mono [&_strong]:text-[22px] [&_strong]:leading-[normal] [&_strong]:font-semibold [&_strong]:text-slate-900 dark:[&_strong]:text-slate-200 [&_span]:text-[10px] [&_span]:text-slate-500">
              <strong>{formatBytes(hardware.totalSystemMemoryBytes)}</strong>
              <span>system memory</span>
            </div>
          </section>

          <section
            className="flex flex-col gap-3.5"
            aria-labelledby="footprint-title"
          >
            <div className="flex items-end justify-between gap-5 [&_h2]:mb-1 [&_h2]:text-[19px] [&_h2]:leading-tight [&_h2]:tracking-[-.015em] [&_h2]:text-slate-900 dark:[&_h2]:text-slate-200 [&_p]:text-xs [&_p]:leading-normal [&_p]:text-slate-600 dark:[&_p]:text-slate-400">
              <div>
                <span className="block text-slate-500 font-sans text-[10px] font-[650] leading-[1.2] tracking-[.09em] uppercase mb-[5px]">
                  Runtime footprint
                </span>
                <h2 id="footprint-title">Primary model</h2>
              </div>
            </div>
            <article className="grid min-h-[68px] grid-cols-[auto_minmax(180px,1fr)_minmax(300px,auto)] items-center gap-3.5 border-t border-slate-300 border-b border-b-slate-200 px-2.5 py-3.5 dark:border-t-slate-750 dark:border-b-slate-800 [&>div]:flex [&>div]:flex-col [&_strong]:text-[13px] [&_strong]:text-slate-900 dark:[&_strong]:text-slate-200 [&_span]:text-[10px] [&_span]:text-slate-500 max-[620px]:grid-cols-[auto_minmax(0,1fr)]">
              <Layers3 size={18} aria-hidden="true" />
              <div>
                <strong>
                  {primary?._tag === "Unassigned" || primary === null
                    ? "No local model selected"
                    : primary._tag === "Resolving"
                    ? "Preparing selected model"
                    : primary.descriptor.displayName}
                </strong>
                <span>
                  {primary
                    ? slotStatus(primary).detail ?? slotStatus(primary).label
                    : "Not configured"}
                </span>
              </div>
              {Result.isSuccess(preview) &&
                primary?._tag === "ConfiguredLocal" &&
                primary.residency._tag === "Unloaded" && (
                  <dl className="grid grid-cols-3 gap-4 [&_div]:min-w-0 [&_dt]:text-[9px] [&_dt]:tracking-[.06em] [&_dt]:uppercase [&_dt]:text-slate-500 [&_dd]:overflow-hidden [&_dd]:text-ellipsis [&_dd]:whitespace-nowrap [&_dd]:text-[11px] [&_dd]:text-slate-600 dark:[&_dd]:text-slate-400 max-[620px]:col-start-2">
                    <div>
                      <dt>Required memory</dt>
                      <dd>
                        {formatBytes(preview.value.requiredSystemMemoryBytes)}
                      </dd>
                    </div>
                    <div>
                      <dt>Context</dt>
                      <dd>
                        {formatContext(preview.value.contextWindowTokens)}
                      </dd>
                    </div>
                    <div>
                      <dt>Parallel</dt>
                      <dd>{preview.value.parallelSequences}</dd>
                    </div>
                  </dl>
                )}
            </article>
          </section>

          <section
            className="flex flex-col gap-3.5"
            aria-labelledby="domains-title"
          >
            <div className="flex items-end justify-between gap-5 [&_h2]:mb-1 [&_h2]:text-[19px] [&_h2]:leading-tight [&_h2]:tracking-[-.015em] [&_h2]:text-slate-900 dark:[&_h2]:text-slate-200 [&_p]:text-xs [&_p]:leading-normal [&_p]:text-slate-600 dark:[&_p]:text-slate-400">
              <div>
                <span className="block text-slate-500 font-sans text-[10px] font-[650] leading-[1.2] tracking-[.09em] uppercase mb-[5px]">
                  Physical topology
                </span>
                <h2 id="domains-title">Memory domains</h2>
                <p>Each allocation is charged once to its physical owner.</p>
              </div>
              <span className="min-w-6 text-right font-mono text-xs leading-[normal] font-semibold text-slate-500">
                {memory?.domains.length ?? 0}
              </span>
            </div>
            <div className="border-t border-slate-300 dark:border-slate-750">
              {memory?.domains.map((domain) => (
                <article
                  className="grid grid-cols-[auto_minmax(0,1fr)] gap-3 border-b border-slate-200 dark:border-slate-800 px-2.5 py-4 text-slate-500"
                  key={domain.id}
                >
                  <MemoryStick size={17} aria-hidden="true" />
                  <div className="min-w-0 [&_p]:mt-2 [&_p]:text-[10px] [&_p]:text-slate-500">
                    <div className="flex justify-between gap-3.5 [&_strong]:text-[12px] [&_strong]:text-slate-900 dark:[&_strong]:text-slate-200 [&_span]:font-mono [&_span]:text-[10px] [&_span]:text-slate-600 dark:[&_span]:text-slate-400">
                      <strong>{domain.label}</strong>
                      <span>
                        {domain.usedBytes === null
                          ? `${formatBytes(domain.totalBytes)} total`
                          : `${formatBytes(domain.usedBytes)} of ${formatBytes(
                              domain.totalBytes
                            )} used`}
                      </span>
                    </div>
                    {domain.freeBytes === null || domain.usedBytes === null ? (
                      <p>{domain.notice}</p>
                    ) : (
                      <>
                        <div
                          className="mt-2.5 flex h-2 overflow-hidden rounded-sm bg-slate-100 dark:bg-slate-800"
                          role="progressbar"
                          aria-label={`${
                            domain.label
                          } memory use: ${formatBytes(
                            domain.fixedBytes ?? 0
                          )} model weights, ${formatBytes(
                            domain.kvCacheBytes ?? 0
                          )} context cache, ${formatBytes(
                            domain.systemAndAppsBytes ?? 0
                          )} system and apps, ${formatBytes(
                            domain.freeBytes
                          )} free`}
                          aria-valuemin={0}
                          aria-valuemax={domain.totalBytes}
                          aria-valuenow={domain.usedBytes}
                        >
                          <span
                            className="bg-blue-700 dark:bg-blue-500"
                            style={{
                              width: `${percentage(
                                domain.fixedBytes ?? 0,
                                domain.totalBytes
                              )}%`,
                            }}
                          />
                          <span
                            className="bg-violet-700 dark:bg-violet-500"
                            style={{
                              width: `${percentage(
                                domain.kvCacheBytes ?? 0,
                                domain.totalBytes
                              )}%`,
                            }}
                          />
                          <span
                            className="bg-slate-500 opacity-55"
                            style={{
                              width: `${percentage(
                                domain.systemAndAppsBytes ?? 0,
                                domain.totalBytes
                              )}%`,
                            }}
                          />
                        </div>
                        <dl className="mt-2.5 grid grid-cols-4 gap-x-[18px] gap-y-2.5 max-[620px]:grid-cols-2 [&_div]:relative [&_div]:min-w-0 [&_div]:pl-[11px] [&_div]:before:absolute [&_div]:before:left-0 [&_div]:before:top-1 [&_div]:before:h-3 [&_div]:before:w-[5px] [&_div]:before:rounded-px [&_dt]:text-[9px] [&_dt]:leading-tight [&_dt]:text-slate-500 [&_dd]:mt-0.5 [&_dd]:font-mono [&_dd]:text-[10px] [&_dd]:leading-[normal] [&_dd]:text-slate-600 dark:[&_dd]:text-slate-400">
                          <div className="before:bg-blue-700 dark:before:bg-blue-500">
                            <dt>Model weights</dt>
                            <dd>{formatBytes(domain.fixedBytes ?? 0)}</dd>
                          </div>
                          <div className="before:bg-violet-700 dark:before:bg-violet-500">
                            <dt>Context cache</dt>
                            <dd>{formatBytes(domain.kvCacheBytes ?? 0)}</dd>
                          </div>
                          <div className="before:bg-slate-500 before:opacity-55">
                            <dt>System &amp; apps</dt>
                            <dd>
                              {formatBytes(domain.systemAndAppsBytes ?? 0)}
                            </dd>
                          </div>
                          <div className="before:border before:border-slate-300 before:bg-slate-100 dark:before:border-slate-750 dark:before:bg-slate-800">
                            <dt>Free</dt>
                            <dd>{formatBytes(domain.freeBytes)}</dd>
                          </div>
                        </dl>
                      </>
                    )}
                  </div>
                </article>
              ))}
            </div>
          </section>

          <section
            className="flex flex-col gap-3.5"
            aria-labelledby="accelerators-title"
          >
            <div className="flex items-end justify-between gap-5 [&_h2]:mb-1 [&_h2]:text-[19px] [&_h2]:leading-tight [&_h2]:tracking-[-.015em] [&_h2]:text-slate-900 dark:[&_h2]:text-slate-200 [&_p]:text-xs [&_p]:leading-normal [&_p]:text-slate-600 dark:[&_p]:text-slate-400">
              <div>
                <span className="block text-slate-500 font-sans text-[10px] font-[650] leading-[1.2] tracking-[.09em] uppercase mb-[5px]">
                  Compute
                </span>
                <h2 id="accelerators-title">Accelerators</h2>
              </div>
              <span className="min-w-6 text-right font-mono text-xs leading-[normal] font-semibold text-slate-500">
                {hardware.accelerators.length}
              </span>
            </div>
            {hardware.accelerators.length === 0 ? (
              <div className="rounded-[10px] border border-dashed border-slate-300 dark:border-slate-750 bg-white dark:bg-slate-850 p-[26px] text-center text-[13px] text-slate-500">
                CPU inference · no accelerator reported
              </div>
            ) : (
              <div className="border-t border-slate-300 dark:border-slate-750">
                {hardware.accelerators.map((accelerator) => (
                  <article
                    className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-3 border-b border-slate-200 dark:border-slate-800 px-2.5 py-[13px] text-slate-500 [&>div]:flex [&>div]:flex-col [&_strong]:text-[12px] [&_strong]:text-slate-900 dark:[&_strong]:text-slate-200 [&_span]:text-[10px] [&_span]:text-slate-500"
                    key={accelerator.acceleratorId}
                  >
                    <Cpu size={17} aria-hidden="true" />
                    <div>
                      <strong>{accelerator.name}</strong>
                      <span>{accelerator.backend}</span>
                    </div>
                    <span>Local inference</span>
                  </article>
                ))}
              </div>
            )}
          </section>
        </>
      )}
    </div>
  )
}
export function ModelSettingsCenter({
  tab,
}: {
  readonly tab: Exclude<SettingsTab, "general" | "archived">
}): ReactNode {
  return (
    <div
      className={`min-w-0 min-h-0 flex-1 ${
        tab === "catalog"
          ? "overflow-hidden max-[840px]:overflow-auto"
          : "overflow-auto"
      }`}
    >
      {tab === "models" ? (
        <ModelsView />
      ) : tab === "catalog" ? (
        <CatalogView />
      ) : (
        <HardwareView />
      )}
    </div>
  )
}
