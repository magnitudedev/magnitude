import type { ReactNode } from "react"
import { Option } from "effect"
import { formatLocalModelDisplayName, localModelIsInstalled, useLocalModelCommandStatus, useLocalModelMutations, type ApplicationPage } from "@magnitudedev/client-common"
import type { CustomLocalModel, CustomModelAcquisitionState, LocalModel } from "@magnitudedev/sdk"
import { HiddenModels, UnhideButton } from "./local-model-lists"
import { FoundModelsBanner, HuggingFaceLink, NoRecommendations } from "./discover-custom"
import { modelLab, type ModelLab } from "./model-logo"
import type { AddDialogRequest } from "./add-model-dialog"

/** A catalog or added model's download and the one-time optimization that follows it. */
export const acquiring = (acquisition: CustomModelAcquisitionState) =>
  acquisition._tag === "Installing" || acquisition._tag === "Updating" || acquisition._tag === "Optimizing"

/**
 * Catalog and My Models' list: `selected` is what the filter and lab selects leave, which the search
 * placeholder counts; `visible` also follows the typed search, in the chosen order.
 */
export const selectModels = ({ library, hidden, filter, lab, search, sort }: {
  library: readonly LocalModel[]
  hidden: readonly LocalModel[]
  filter: string
  lab: ModelLab | null
  search: string
  sort: string
}): { readonly selected: readonly LocalModel[]; readonly visible: readonly LocalModel[] } => {
  const matchesLab = (model: LocalModel) => lab === null || modelLab(model) === lab
  const selected: readonly LocalModel[] = filter === "hidden" ? hidden.filter(matchesLab) : library.filter(model => {
    const matchesFilter = filter === "all"
      || filter === "fits" && model.servingState._tag === "Assessed" && model.servingState.assessment._tag === "Fits"
      || filter === "downloaded" && localModelIsInstalled(model)
      || filter === "downloading" && acquiring(model.acquisitionState)
    return matchesFilter && matchesLab(model)
  })
  const query = search.trim().toLowerCase()
  const visible = selected.filter(model => `${formatLocalModelDisplayName(model)} ${model._tag === "Catalog" ? model.presentation.description : model.presentation.sourceLabel}`.toLowerCase().includes(query))
  if (sort !== "recommended") visible.sort((a, b) => {
    const byName = formatLocalModelDisplayName(a).localeCompare(formatLocalModelDisplayName(b), undefined, { numeric: true }) || a.modelId.localeCompare(b.modelId)
    return sort === "smallest" ? a.storageBytes - b.storageBytes || byName : sort === "largest" ? b.storageBytes - a.storageBytes || byName : byName
  })
  return { selected, visible }
}

/** Find other versions on Hugging Face: only a Catalog page's catalog cards, searching for the model's name. */
export const otherVersionsRequest = (page: ApplicationPage, model: LocalModel): Option.Option<AddDialogRequest> =>
  page === "catalog" && model._tag === "Catalog" ? Option.some({ _tag: "Search", query: model.presentation.displayName }) : Option.none()

/** Adding a model closes the dialog and shows it in My Models, where its download progresses. */
export const afterModelAdded = (closeDialog: () => void, navigate: (page: ApplicationPage) => void) => () => {
  closeDialog()
  navigate("models")
}

/** Unhide for one hidden model, disabled while any command for it is pending. */
export function HiddenModelUnhide({ model }: { model: CustomLocalModel }) {
  const { unhide } = useLocalModelMutations()
  const command = useLocalModelCommandStatus(model.modelId)
  return <UnhideButton pending={command.pending} onUnhide={() => unhide(model.modelId)} />
}

/** The cards under Catalog and My Models' header; the Hidden filter lists hidden models with Unhide instead. */
export function ModelsListing({ filter, models, card, empty, footer }: {
  filter: string
  models: readonly LocalModel[]
  card: (model: LocalModel) => ReactNode
  empty: ReactNode
  footer?: ReactNode
}) {
  if (filter === "hidden") return <HiddenModels models={models.filter((model): model is CustomLocalModel => model._tag === "Custom")} unhide={model => <HiddenModelUnhide model={model} />} />
  return <>
    <div className="grid items-start gap-5">{models.map(card)}</div>
    {models.length === 0 && empty}
    {footer}
  </>
}

/** Whether Discover's recommendations are still coming, ranked, none fit, or can't be ranked without the hardware. */
export type DiscoverRecommendationState = "pending" | "ranked" | "none" | "unavailable"

/**
 * Discover: the hardware card, the found-models banner, the recommendations, then one Hugging Face
 * line below them; when nothing fits, Add from Hugging Face sits beside Explore Catalog instead.
 */
export function DiscoverSections({ hardware, found, notice, preference, recommendations, state, onViewFound, onAdd, onExplore }: {
  hardware: ReactNode
  found: readonly CustomLocalModel[]
  notice?: ReactNode
  preference: ReactNode
  recommendations: ReactNode
  state: DiscoverRecommendationState
  onViewFound: () => void
  onAdd: () => void
  onExplore: () => void
}) {
  return <>
    {hardware}
    <FoundModelsBanner models={found} onView={onViewFound} />
    {notice}
    {preference}
    {recommendations}
    {state === "ranked" && <HuggingFaceLink onAdd={onAdd} />}
    {state === "none" && <NoRecommendations onExplore={onExplore} onAdd={onAdd} />}
  </>
}
