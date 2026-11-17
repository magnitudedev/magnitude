import { useContext, useEffect, useMemo, useState } from "react"
import { Equal, Option } from "effect"
import { Atom, RegistryContext, Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { CheckIcon, CircleNotchIcon, DownloadSimpleIcon, LightningIcon } from "@phosphor-icons/react"
import { HUGGING_FACE_SEARCH_QUERY_MAXIMUM_BYTES, repositoryDisplayName, type CatalogFormModelId, type CustomFormModelId, type DrafterChoice, type HuggingFaceRepositoryCheck, type HuggingFaceRepositoryId, type HuggingFaceSearchResponse, type HuggingFaceSearchResult, type LocalModel } from "@magnitudedev/sdk"
import { drafterCandidateDownloadBytes, formatStorageSize, readSearchChecks, useAgentClient, useLocalModelMutations, useLocalModels, type AgentClient, type QuickSearchCheck, type ServingMemory } from "@magnitudedev/client-common"
import { Button } from "../ui/button"
import { labForDeveloper } from "../model-logo"
import { downloadContents, methodName, onThisComputer, quantView, transferOf } from "./data"
import { AUTO_DRAFTER, DrafterSection, drafterChoice, useDrafters, type DrafterSelection } from "./drafter-picker"
import type { QuantView } from "./view"
import type { OfferedModel } from "./expanding-model"
import { ExploreView, type ExploreResults, type ExploreRow, type RunnableFamily } from "./explore-view"

function useDebounced<A>(value: A, delay: number): A {
  const [settled, setSettled] = useState(value)
  useEffect(() => {
    const timer = setTimeout(() => setSettled(value), delay)
    return () => clearTimeout(timer)
  }, [value, delay])
  return settled
}

/** What a search check reads: the repository the search listed, as it listed it. */
const checkInput = (result: HuggingFaceSearchResult) => ({
  repository: result.repository,
  commit: result.commit,
  architecture: result.architecture,
  restricted: result.gated || result.private,
})

/** Whether a check found the repository's best quant here, found it doesn't run or holds no model, or found neither. */
const complete = (check: HuggingFaceRepositoryCheck) =>
  check._tag === "NotAModel" || Option.isSome(check.unsupported) || Option.isSome(check.recommended)

/** A quick check's state as the search schedule reads it: running until it ends, and runnable when it offers a quant. */
const quickState = (check: Result.Result<HuggingFaceRepositoryCheck, unknown>): QuickSearchCheck =>
  Result.isInitial(check) || check.waiting ? "Running"
    : Result.isSuccess(check) && check.value._tag === "Model" && Option.isNone(check.value.unsupported) && Option.isSome(check.value.recommended) ? "Runnable"
    : "Settled"

/**
 * A search result as its check finds it: still checking (or not yet reached); nothing when it doesn't run here or holds
 * no model (search shows what runs); unchecked, to try again, when the check failed or ran out of time (a gated
 * repository's files can't be read until its terms are accepted); else the
 * repository offering the quant it recommends for this machine until it is opened, fully checked in the background
 * when `prefetched`.
 */
/** What a search's schedule has read: each listed repository's quick check it has read, and those it fully checked. */
export interface SearchSchedule {
  readonly checks: ReadonlyMap<string, Result.Result<HuggingFaceRepositoryCheck, unknown>>
  readonly full: ReadonlySet<string>
}

/**
 * Whether two passes of a search's schedule read the same checks to the same results and fully checked the same
 * repositories. A pass runs whenever any check it reads changes at all (its staleness, a full check's progress); only
 * a pass that found something new is a new schedule, so the list is drawn again only then.
 */
export const sameSchedule = (previous: SearchSchedule, next: SearchSchedule): boolean =>
  previous.checks.size === next.checks.size
  && [...next.checks].every(([key, check]) => previous.checks.has(key) && Equal.equals(previous.checks.get(key), check))
  && previous.full.size === next.full.size
  && [...next.full].every(key => previous.full.has(key))

/**
 * A search's schedule over its `listed` repositories, `visible` being those on screen: each pass reads the checks
 * `readSearchChecks` runs, and stays the same schedule unless it found something new.
 */
export const searchSchedule = (client: Pick<AgentClient, "Models">, listed: readonly HuggingFaceSearchResult[], visible: Atom.Atom<ReadonlySet<string>>) =>
  Atom.make((get): SearchSchedule => {
    const byKey = new Map<string, HuggingFaceSearchResult>(listed.map(result => [result.repository, result]))
    const checks = new Map<string, Result.Result<HuggingFaceRepositoryCheck, unknown>>()
    const full = new Set<string>()
    const previous = get.self<SearchSchedule>()
    readSearchChecks({
      listed: listed.map(result => result.repository),
      visible: get(visible),
      read: new Set(Option.match(previous, { onNone: () => [], onSome: value => [...value.checks.keys()] })),
      quick: (key) => {
        const check = get(client.Models.CheckHuggingFaceRepository(checkInput(byKey.get(key)!))).result
        checks.set(key, check)
        return quickState(check)
      },
      full: (key) => {
        full.add(key)
        const inspected = get(client.Models.InspectModelCandidate({ reference: key })).result
        return !Result.isInitial(inspected) && !inspected.waiting
      },
    })
    const next = { checks, full }
    return Option.match(previous, { onNone: () => next, onSome: value => sameSchedule(value, next) ? value : next })
  })

const searchRow = (result: HuggingFaceSearchResult, check: Result.Result<HuggingFaceRepositoryCheck, unknown>, prefetched: boolean, retry: () => void, memory: ServingMemory): ExploreRow | null => {
  const developer = Option.getOrNull(result.developer)
  const named = { key: result.repository, name: result.displayName, lab: developer ? labForDeveloper(developer.lab) : undefined }
  if (quickState(check) === "Running") return { _tag: "Checking", ...named }
  if (!Result.isSuccess(check) || !complete(check.value)) return { _tag: "Unchecked", ...named, gated: result.gated, retry }
  const checked = check.value
  if (checked._tag === "NotAModel" || Option.isSome(checked.unsupported)) return null
  return Option.match(checked.recommended, {
    onNone: () => null,
    onSome: (version): ExploreRow => {
      const base = Option.getOrNull(Option.orElse(result.base, () => checked.base))
      const lab = Option.getOrNull(Option.orElse(checked.developer, () => result.developer))
      const quant = quantView({ key: version.modelId, name: version.label, repository: result.repository, sizeBytes: version.sizeBytes, downloadBytes: version.downloadBytes }, version.assessment, memory)
      return {
        _tag: "Offered",
        model: {
          key: result.repository,
          catalog: false,
          name: result.displayName,
          lab: lab ? labForDeveloper(lab.lab) : undefined,
          country: lab?.country ?? null,
          fineTuneOf: base && base.relation === "Finetune" ? repositoryDisplayName(base.repository) : null,
          drafter: null,
          contextLimit: version.assessment._tag === "Fits" ? version.assessment.profile.contextLength : null,
          quants: [quant],
          recommended: quant.key,
          checking: false,
        },
        prefetched,
      }
    },
  })
}

/**
 * An opened search result with every quant its repository has, smallest first, each as this machine runs it, from the service's
 * inspection of the repository; until that is done, the quant search offered, marked as still checking. A catalog
 * model, or one not open, is as it is.
 */
const useRepositoryQuants = (memory: ServingMemory) => function useOpened(model: OfferedModel, open: boolean): OfferedModel {
  const client = useAgentClient()
  const reference = !model.catalog && open ? model.key : null
  const inspected = useAtomValue(useMemo(() => Atom.make((get) => reference === null
    ? Result.initial()
    : get(client.Models.InspectModelCandidate({ reference })).result),
  [client, reference]))
  if (reference === null) return model
  if (!Result.isSuccess(inspected)) return { ...model, checking: !Result.isFailure(inspected) }
  const repository = model.key
  const source = inspected.value.source
  const quants = inspected.value.versions.map(version => quantView(
    { key: version.modelId, name: version.label, repository, sizeBytes: version.sizeBytes, downloadBytes: version.downloadBytes, drafterTarget: { _tag: "Candidate", source, files: version.files } },
    Option.getOrNull(version.assessment), memory)).sort((left, right) => left.sizeBytes - right.sizeBytes)
  if (quants.length === 0) return model
  const best = inspected.value.versions.find(version => version.recommended)
  const drafter = Option.getOrNull(Option.flatMap(Option.fromNullable(best), version => version.drafter))
  return {
    ...model,
    quants,
    recommended: best?.modelId ?? model.recommended,
    drafter: drafter ? { method: Option.match(drafter.method, { onNone: () => drafter.name, onSome: method => methodName[method._tag] }), sizeBytes: drafterCandidateDownloadBytes(drafter) } : null,
    checking: false,
  }
}

/** What leads Explore's results: a recommended model, offered from its ladder, or a repository looked up by name. */
export type ExploreFirst = { readonly _tag: "Offered"; readonly model: OfferedModel } | { readonly _tag: "Repository"; readonly repository: string }

/**
 * What a search for a model's quants leads with: the recommended model published in `repository`, offered from its
 * ladder as it is recommended, since Hugging Face search leaves recommended models' repositories out; else the
 * repository, looked up by name.
 */
export const exploreFirst = (repository: string, offered: readonly { readonly model: OfferedModel }[]): ExploreFirst => {
  const model = offered.find(entry => entry.model.quants.some(quant => quant.repository === repository))?.model
  return model ? { _tag: "Offered", model } : { _tag: "Repository", repository }
}

interface ExploreProps {
  readonly query: string
  readonly first: ExploreFirst | null
  readonly setQuery: (value: string) => void
  readonly memory: ServingMemory
  readonly families: readonly RunnableFamily[]
  readonly owned: ReadonlySet<string>
}

/**
 * A quant's Download, which says it is starting from the click until the service lists the download (this button is
 * then replaced); if nothing is listed within half a minute, the request failed and it offers Download again.
 */
function DownloadButton({ size, contents, disabled, start }: { size: number; contents: string | null; disabled: boolean; start: () => void }) {
  const [starting, setStarting] = useState(false)
  useEffect(() => {
    if (!starting) return
    const timer = setTimeout(() => setStarting(false), 30_000)
    return () => clearTimeout(timer)
  }, [starting])
  if (starting) return <Button disabled><CircleNotchIcon className="motion-safe:animate-spin" />Starting…</Button>
  return <span className="flex items-center gap-3">
    {contents && <span className="text-xs text-slate-500">{contents}</span>}
    <Button disabled={disabled} onClick={() => { setStarting(true); start() }}><DownloadSimpleIcon />Download {formatStorageSize(size)}</Button>
  </span>
}

/** A Hugging Face quant's Download: its files and projector, plus the drafter chosen for it unless that is here already. */
function HuggingFaceDownload({ quant, repository, selection, start }: {
  quant: QuantView; repository: HuggingFaceRepositoryId; selection: DrafterSelection; start: (drafter: DrafterChoice) => void
}) {
  const target = quant.drafterTarget
  const { chosen } = useDrafters(target ?? { _tag: "Model", modelId: quant.key as CustomFormModelId }, selection)
  const drafterBytes = target ? Option.match(chosen, { onNone: () => 0, onSome: drafterCandidateDownloadBytes }) : 0
  return <DownloadButton key={`${repository}:${quant.key}`} size={quant.downloadBytes + drafterBytes} contents={downloadContents({ drafter: null }, quant)} disabled={!quant.fits}
    start={() => start(target ? drafterChoice(selection, chosen) : { _tag: "Auto" })} />
}

/** The footer of an opened model: its quant's Download, or how far that download is, or that it's here. */
export function useDownload(selections: Readonly<Record<string, DrafterSelection>>) {
  const client = useAgentClient()
  const add = useAtomSet(client.Models.AddHuggingFaceModel)
  const { install } = useLocalModelMutations()
  const local = useLocalModels()
  const byId = new Map<string, LocalModel>(Result.isSuccess(local) ? local.value.models.map(model => [model.modelId, model]) : [])
  return (model: OfferedModel, quant: QuantView) => {
    const existing = byId.get(quant.key)
    const transfer = existing ? transferOf(existing) : undefined
    if (transfer) {
      const percent = transfer.totalBytes > 0 ? Math.round(transfer.completedBytes / transfer.totalBytes * 100) : 0
      return <Button disabled><CircleNotchIcon className="motion-safe:animate-spin" />Downloading {percent}%</Button>
    }
    if (existing && onThisComputer(existing)) return <Button disabled variant="outline"><CheckIcon />Downloaded</Button>
    if (model.catalog) return <DownloadButton key={quant.key} size={quant.downloadBytes} contents={downloadContents({ drafter: null }, quant)} disabled={!quant.fits || quant.missingDiskBytes > 0} start={() => install(quant.key as CatalogFormModelId)} />
    if (quant.repository === null) return null
    const repository = quant.repository as HuggingFaceRepositoryId
    return <HuggingFaceDownload key={quant.key} quant={quant} repository={repository} selection={selections[quant.key] ?? AUTO_DRAFTER}
      start={drafter => add({ repository, revision: Option.none(), version: Option.some(quant.name), drafter, force: false })} />
  }
}

/** A catalog model's drafter, which comes with it: named beside its download, with the bytes it adds. */
function CatalogDrafter({ drafter }: { drafter: NonNullable<OfferedModel["drafter"]> }) {
  return <span className="inline-flex items-center gap-1.5 text-xs text-slate-600 dark:text-slate-400" title="A drafter is a small model that drafts tokens ahead to speed up replies.">
    <LightningIcon weight="fill" className="size-3.5 text-amber-500" />
    <span>Using <span className="font-medium text-slate-800 dark:text-slate-200">{drafter.method} drafter</span></span>
    {drafter.sizeBytes > 0 && <span className="text-slate-500">· {formatStorageSize(drafter.sizeBytes)}</span>}
  </span>
}

/**
 * The drafter beside an opened model's download: a Hugging Face quant's choice once its repository is inspected, a
 * catalog quant's own drafter, or nothing for a catalog quant without one.
 */
const drafterFor = (selections: Readonly<Record<string, DrafterSelection>>, select: (key: string, selection: DrafterSelection) => void) =>
  (model: OfferedModel, quant: QuantView) => model.catalog
    ? model.drafter && quant.drafterBytes > 0 ? <CatalogDrafter drafter={model.drafter} /> : null
    : quant.drafterTarget
      ? <DrafterSection key={quant.key} target={quant.drafterTarget} selection={selections[quant.key] ?? AUTO_DRAFTER} onSelect={selection => select(quant.key, selection)} />
      : <span role="status" className="flex items-center gap-1.5 text-xs text-slate-500"><CircleNotchIcon aria-hidden="true" className="size-3.5 motion-safe:animate-spin" />Finding a drafter…</span>

/**
 * Search results under `first`, a repository looked up on its own, so it leads whether or not the search found it;
 * the rest follow in their order, without it again.
 */
export const listedFirst = (results: readonly HuggingFaceSearchResult[], first: HuggingFaceSearchResult | undefined): readonly HuggingFaceSearchResult[] =>
  first === undefined ? results : [first, ...results.filter(result => result.repository !== first.repository)]

/**
 * A repository looked up by its own name: a search for exactly it, which Hugging Face answers with it, so a model's
 * own repository can lead a search for its other quants even where that search would rank it out of reach.
 */
const useRepositoryResult = (repository: string | null, settled: string): HuggingFaceSearchResult | undefined => {
  const client = useAgentClient()
  const found = useAtomValue(useMemo(() => Atom.make((get) => repository === null || settled === ""
    ? Result.initial<HuggingFaceSearchResponse>()
    : get(client.Models.SearchHuggingFaceModels({ query: repository, limit: Option.some(5), cursor: Option.none() })).result),
  [client, repository, settled]))
  return Result.isSuccess(found) ? found.value.results.find(result => result.repository === repository) : undefined
}

/** No results yet: one list, so what is derived from a search's results doesn't change before they do. */
const NO_RESULTS: readonly HuggingFaceSearchResult[] = []

/**
 * The settled query's Hugging Face results, a page at a time: pages already shown stay while the next one loads, and
 * reaching the end of the list asks for it. An empty query asks for nothing. Its repositories are checked in the order
 * `readSearchChecks` runs them, `visible` being those on screen.
 */
function useSearchResults(settled: string, query: string, first: ExploreFirst | null, memory: ServingMemory, visible: Atom.Writable<ReadonlySet<string>>): ExploreResults {
  const client = useAgentClient()
  const [pages, setPages] = useState<{ readonly query: string; readonly cursor: Option.Option<string>; readonly shown: readonly HuggingFaceSearchResult[] }>(
    { query: settled, cursor: Option.none(), shown: [] })
  const current = pages.query === settled ? pages : { query: settled, cursor: Option.none<string>(), shown: NO_RESULTS }
  const page = useAtomValue(useMemo(() => Atom.make((get) => settled === ""
    ? Result.initial()
    : get(client.Models.SearchHuggingFaceModels({ query: settled, limit: Option.none(), cursor: current.cursor })).result),
  [client, settled, current.cursor]))
  const loading = Result.isInitial(page) || page.waiting
  const listed = Result.isSuccess(page) && !page.waiting ? page.value : null
  const next = listed ? listed.nextCursor : Option.none<string>()
  const more = () => {
    if (!listed || Option.isNone(next)) return
    setPages({ query: settled, cursor: next, shown: [...current.shown, ...listed.results] })
  }
  const pinned = useRepositoryResult(first?._tag === "Repository" ? first.repository : null, settled)
  const listedResults = useMemo(() => listedFirst([...current.shown, ...(listed?.results ?? [])], pinned), [current.shown, listed, pinned])
  const schedule = useAtomValue(useMemo(() => searchSchedule(client, listedResults, visible), [client, listedResults, visible]))
  const registry = useContext(RegistryContext)
  return query !== settled || (loading && current.shown.length === 0) ? { _tag: "Searching" }
    : Result.isFailure(page) && current.shown.length === 0 ? { _tag: "Failed", message: "Hugging Face search failed. Check your connection and try again." }
    : {
        _tag: "Results",
        rows: [...(first?._tag === "Offered" ? [{ _tag: "Offered" as const, model: first.model, prefetched: false }] : []), ...listedResults.flatMap((result) => {
          const retry = () => registry.refresh(client.Models.CheckHuggingFaceRepository(checkInput(result)))
          const row = searchRow(result, schedule.checks.get(result.repository) ?? Result.initial(), schedule.full.has(result.repository), retry, memory)
          return row ? [row] : []
        })],
        more: loading ? "Loading" : Option.isSome(next) ? more : null,
      }
}

/** Explore models on the service's data: a Hugging Face search, and what to show before one. */
export function Explore(props: ExploreProps) {
  const settled = useDebounced(props.query.trim(), 350)
  const tooLong = new TextEncoder().encode(settled).length > HUGGING_FACE_SEARCH_QUERY_MAXIMUM_BYTES
  const visible = useMemo(() => Atom.make<ReadonlySet<string>>(new Set<string>()), [])
  const setVisible = useAtomSet(visible)
  const searched = useSearchResults(tooLong ? "" : settled, props.query.trim(), props.first, props.memory, visible)
  const [selections, setSelections] = useState<Readonly<Record<string, DrafterSelection>>>({})
  const download = useDownload(selections)
  const drafters = drafterFor(selections, (key, selection) => setSelections(current => ({ ...current, [key]: selection })))
  const results: ExploreResults = props.query.trim() === ""
    ? { _tag: "Idle" }
    : tooLong ? { _tag: "Failed", message: "This search is longer than Hugging Face accepts. Shorten it." }
    : searched
  const useOpened = useMemo(() => useRepositoryQuants(props.memory), [props.memory])
  return <ExploreView {...props} results={results} download={download} drafters={drafters} useOpened={useOpened} onSeen={setVisible} />
}
