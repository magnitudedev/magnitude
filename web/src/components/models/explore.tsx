import { useContext, useEffect, useMemo, useState } from "react"
import { Equal, Option } from "effect"
import { Atom, RegistryContext, Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { CheckIcon, CircleNotchIcon, DownloadSimpleIcon, LightningIcon } from "@phosphor-icons/react"
import { HUGGING_FACE_SEARCH_QUERY_MAXIMUM_BYTES, repositoryDisplayName, type CatalogFormModelId, type DrafterChoice, type HuggingFaceRepositoryCheck, type HuggingFaceRepositoryId, type HuggingFaceSearchResponse, type HuggingFaceSearchResult, type LocalModel } from "@magnitudedev/sdk"
import { SEARCH_BATCH_SIZE, drafterCandidateDownloadBytes, formatStorageSize, readSearchChecks, useAgentClient, useLocalModelMutations, useLocalModels, type AgentClient, type QuickSearchCheck, type SearchChecksProgress, type ServingMemory } from "@magnitudedev/client-common"
import { Button } from "../ui/button"
import { labForDeveloper } from "../model-logo"
import { downloadContents, transferLabel, methodName, onThisComputer, quantView, transferOf } from "./data"
import { AUTO_DRAFTER, DrafterSection, drafterChoice, useDrafters, type DrafterSelection } from "./drafter-picker"
import type { QuantView } from "./view"
import type { OfferedModel } from "./expanding-model"
import { ExploreView, type ExploreMore, type ExploreResults, type ExploreRow, type RunnableFamily } from "./explore-view"

/** Consecutive pages with zero runnable models before auto-pagination pauses for an explicit click. */
const MAX_AUTO_EMPTY_PAGES = 2

/**
 * How long already-settled models wait behind an earlier still-checking repository to preserve strict download order.
 * Once a single head check takes longer than this, ready models behind it are released and locked in place so the user
 * never stares at blank skeletons for 10–30 seconds; if the slow check later succeeds, it appends below without jumping.
 */
export const FRONTIER_GRACE_MS = 1500

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

/** Candidate inspection reference pinned to the searched commit so inspection reuses the cached commit snapshot. */
export const inspectReference = (repository: string, commit?: string | null) =>
  commit ? `${repository}@${commit}` : repository

/** Whether a check found the repository's best quant here, found it doesn't run or holds no model, or found neither. */
const complete = (check: HuggingFaceRepositoryCheck) =>
  check._tag === "NotAModel" || Option.isSome(check.unsupported) || Option.isSome(check.recommended)

/**
 * A quick check's state as the search schedule reads it:
 * - `"Checking"` before its first completion
 * - `"Retrying"` while a previously shown unchecked row is being retried
 * - `"Hidden"` when it doesn't run here, holds no model, or failed on a non-actionable search row
 * - `"Runnable"` when it offers a recommended quant
 * - `"Unchecked"` when the check didn't complete and `showUnchecked` is true (gated repo or direct single-repo lookup)
 */
export const quickState = (
  check: Result.Result<HuggingFaceRepositoryCheck, unknown>,
  showUnchecked = true,
): QuickSearchCheck => {
  if (Result.isInitial(check)) return "Checking"
  if (check.waiting) return showUnchecked ? "Retrying" : "Checking"
  if (!Result.isSuccess(check) || !complete(check.value)) return showUnchecked ? "Unchecked" : "Hidden"
  if (check.value._tag === "NotAModel" || Option.isSome(check.value.unsupported)) return "Hidden"
  return Option.isSome(check.value.recommended) ? "Runnable" : "Hidden"
}

/**
 * Appends newly settled visible repositories to `previous` without ever reordering or removing already-revealed rows.
 * Normally only the contiguous settled prefix `0 .. frontier - 1` is released (exact download order); if
 * `releaseBehindFrontier` is true because the frontier repository stalled past `FRONTIER_GRACE_MS`, ready repositories
 * behind it are also appended in their download order.
 */
export const advanceRevealed = (input: {
  readonly previous: readonly string[]
  readonly listed: readonly HuggingFaceSearchResult[]
  readonly checks: ReadonlyMap<string, Result.Result<HuggingFaceRepositoryCheck, unknown>>
  readonly frontier: number
  readonly releaseBehindFrontier: boolean
  readonly showUnchecked: (result: HuggingFaceSearchResult) => boolean
}): readonly string[] => {
  const existing = new Set(input.previous)
  const limit = input.releaseBehindFrontier ? input.listed.length : input.frontier
  let next: string[] | null = null
  for (let index = 0; index < limit; index += 1) {
    const result = input.listed[index]!
    if (existing.has(result.repository)) continue
    const check = input.checks.get(result.repository)
    if (!check) continue
    const state = quickState(check, input.showUnchecked(result))
    if (state === "Runnable" || state === "Unchecked" || state === "Retrying") {
      if (next === null) next = [...input.previous]
      next.push(result.repository)
      existing.add(result.repository)
    }
  }
  return next ?? input.previous
}

/** What a search's schedule has read: each listed repository's quick check, those fully checked, the locked reveal order, and frontier progress. */
export interface SearchSchedule {
  readonly checks: ReadonlyMap<string, Result.Result<HuggingFaceRepositoryCheck, unknown>>
  readonly full: ReadonlySet<string>
  readonly revealed: readonly string[]
  readonly progress: SearchChecksProgress
}

/**
 * Whether two passes of a search's schedule read the same checks to the same results, fully checked the same
 * repositories, revealed the same rows, and reached the same progress. A pass runs whenever any check it reads changes
 * at all (its staleness, a full check's progress); only a pass that found something new is a new schedule, so the list
 * is drawn again only then.
 */
export const sameSchedule = (previous: SearchSchedule, next: SearchSchedule): boolean =>
  previous.progress.frontier === next.progress.frontier
  && previous.progress.shown === next.progress.shown
  && previous.progress.checking === next.progress.checking
  && previous.progress.exhausted === next.progress.exhausted
  && previous.revealed.length === next.revealed.length
  && next.revealed.every((key, index) => previous.revealed[index] === key)
  && previous.checks.size === next.checks.size
  && [...next.checks].every(([key, check]) => previous.checks.has(key) && Equal.equals(previous.checks.get(key), check))
  && previous.full.size === next.full.size
  && [...next.full].every(key => previous.full.has(key))

/**
 * A search's schedule over its `listed` repositories up to `target` visible rows: each pass reads the checks
 * `readSearchChecks` runs strictly in list order, appends ready rows into a locked append-only order, and stays the
 * same schedule unless it found something new.
 */
export const searchSchedule = (
  client: Pick<AgentClient, "Models">,
  listed: Atom.Atom<readonly HuggingFaceSearchResult[]>,
  target: Atom.Atom<number>,
  options?: {
    readonly releasedFrontier?: Atom.Atom<string | null>
    readonly pinnedRepository?: string | null
    readonly opened?: Atom.Atom<string | null>
    readonly hovered?: Atom.Atom<string | null>
  },
) =>
  Atom.make((get): SearchSchedule => {
    const items = get(listed)
    const byKey = new Map<string, HuggingFaceSearchResult>(items.map(result => [result.repository, result]))
    const checks = new Map<string, Result.Result<HuggingFaceRepositoryCheck, unknown>>()
    const full = new Set<string>()
    const previous = get.self<SearchSchedule>()
    const previousRevealed = Option.match(previous, { onNone: () => [] as readonly string[], onSome: value => value.revealed })
    const urgent = options?.opened ? get(options.opened) : null
    const hovered = options?.hovered ? get(options.hovered) : null
    // Only show an unchecked/retry card when the user can act on it (gated repo) or explicitly looked up that one repo;
    // transient failures/timeouts on general search rows are skipped instead of clogging the top of the list.
    const showUnchecked = (result: HuggingFaceSearchResult) =>
      result.gated || result.repository === options?.pinnedRepository || items.length === 1
    const progress = readSearchChecks({
      listed: items.map(result => result.repository),
      target: get(target),
      revealed: new Set(previousRevealed),
      prefetched: Option.match(previous, { onNone: () => undefined, onSome: value => value.full }),
      urgent,
      priority: hovered ? [hovered] : undefined,
      quick: (key) => {
        const result = byKey.get(key)!
        const check = get(client.Models.CheckHuggingFaceRepository(checkInput(result))).result
        checks.set(key, check)
        return quickState(check, showUnchecked(result))
      },
      full: (key) => {
        full.add(key)
        const result = byKey.get(key)
        const inspected = get(client.Models.InspectModelCandidate({ reference: inspectReference(key, result?.commit) })).result
        return !Result.isInitial(inspected) && !inspected.waiting
      },
    })
    const releasedKey = options?.releasedFrontier ? get(options.releasedFrontier) : null
    const frontierKey = progress.frontier < items.length ? items[progress.frontier]!.repository : null
    const revealed = advanceRevealed({
      previous: previousRevealed,
      listed: items,
      checks,
      frontier: progress.frontier,
      releaseBehindFrontier: frontierKey !== null && releasedKey === frontierKey,
      showUnchecked,
    })
    const next: SearchSchedule = { checks, full, revealed, progress }
    return Option.match(previous, { onNone: () => next, onSome: value => sameSchedule(value, next) ? value : next })
  })

/**
 * A settled search result as its row shows it: nothing when it doesn't run here or holds no model (search shows what
 * runs); unchecked, to try again, when gated or directly looked up; else the repository offering the quant it
 * recommends for this machine until it is opened, fully checked in the background when `prefetched`.
 */
const searchRow = (result: HuggingFaceSearchResult, check: Result.Result<HuggingFaceRepositoryCheck, unknown>, prefetched: boolean, showUnchecked: boolean, retry: () => void, memory: ServingMemory): ExploreRow | null => {
  const state = quickState(check, showUnchecked)
  if (state === "Checking" || state === "Hidden") return null
  const developer = Option.getOrNull(result.developer)
  const named = { key: result.repository, name: result.displayName, lab: developer ? labForDeveloper(developer.lab) : undefined }
  if (state === "Unchecked" || state === "Retrying") return { _tag: "Unchecked", ...named, gated: result.gated, retrying: state === "Retrying", retry }
  if (!Result.isSuccess(check) || check.value._tag !== "Model") return null
  const checked = check.value
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
          commit: result.commit,
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
          downloads: result.downloads,
          uploader: result.uploader,
          quantCount: Option.getOrNull(result.versionCount),
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
  const reference = !model.catalog && open ? inspectReference(model.key, model.commit) : null
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
    quantCount: quants.length,
    recommended: best?.modelId ?? model.recommended,
    drafter: drafter ? { method: Option.match(drafter.method, { onNone: () => drafter.name, onSome: method => methodName[method._tag] }), sizeBytes: drafterCandidateDownloadBytes(drafter) } : null,
    checking: false,
  }
}

/** What leads Explore's results: a recommended model, offered from its ladder, or a repository looked up by name. */
export type ExploreFirst = { readonly _tag: "Offered"; readonly model: OfferedModel } | { readonly _tag: "Repository"; readonly repository: string }

/**
 * What a search for a model's other quants leads with: the recommended model published in `repository`, offered from its
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

/** A Hugging Face quant's Download once its candidate target is known, adding the chosen drafter's download size. */
function HuggingFaceDownloadWithDrafter({ quant, target, repository, selection, start }: {
  quant: QuantView; target: NonNullable<QuantView["drafterTarget"]>; repository: HuggingFaceRepositoryId; selection: DrafterSelection; start: (drafter: DrafterChoice) => void
}) {
  const { chosen } = useDrafters(target, selection)
  const drafterBytes = Option.match(chosen, { onNone: () => 0, onSome: drafterCandidateDownloadBytes })
  return <DownloadButton key={`${repository}:${quant.key}`} size={quant.downloadBytes + drafterBytes} contents={downloadContents({ drafter: null }, quant)} disabled={!quant.fits}
    start={() => start(drafterChoice(selection, chosen))} />
}

/** A Hugging Face quant's Download: its files and projector, plus the drafter chosen for it once inspected. */
function HuggingFaceDownload({ quant, repository, selection, start }: {
  quant: QuantView; repository: HuggingFaceRepositoryId; selection: DrafterSelection; start: (drafter: DrafterChoice) => void
}) {
  if (quant.drafterTarget !== null) {
    return <HuggingFaceDownloadWithDrafter quant={quant} target={quant.drafterTarget} repository={repository} selection={selection} start={start} />
  }
  return <DownloadButton key={`${repository}:${quant.key}`} size={quant.downloadBytes} contents={downloadContents({ drafter: null }, quant)} disabled={!quant.fits}
    start={() => start({ _tag: "Auto" })} />
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
    if (transfer) return <Button disabled><CircleNotchIcon className="motion-safe:animate-spin" />{transferLabel(transfer).text}</Button>
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
      : model.checking
        ? <span role="status" className="flex items-center gap-1.5 text-xs text-slate-500"><CircleNotchIcon aria-hidden="true" className="size-3.5 motion-safe:animate-spin" />Finding a drafter…</span>
        : null

/** Removes duplicate repositories across live-paginated results, keeping the first (highest-ranked) occurrence. */
export const dedupeResults = (results: readonly HuggingFaceSearchResult[]): readonly HuggingFaceSearchResult[] => {
  const seen = new Set<string>()
  const unique: HuggingFaceSearchResult[] = []
  for (const result of results) {
    if (seen.has(result.repository)) continue
    seen.add(result.repository)
    unique.push(result)
  }
  return unique
}

/**
 * Search results under `first`, a repository looked up on its own, so it leads whether or not the search found it;
 * the rest follow in their order, without it again.
 */
export const listedFirst = (results: readonly HuggingFaceSearchResult[], first: HuggingFaceSearchResult | undefined): readonly HuggingFaceSearchResult[] =>
  first === undefined ? results : [first, ...results.filter(result => result.repository !== first.repository)]

/** No results yet: one list, so what is derived from a search's results doesn't change before they do. */
const NO_RESULTS: readonly HuggingFaceSearchResult[] = []

interface SearchPaging {
  readonly query: string
  readonly cursor: Option.Option<string>
  readonly shown: readonly HuggingFaceSearchResult[]
  readonly target: number
  readonly emptyPages: number
  readonly shownBeforePage: number
}

const initialPaging = (query: string): SearchPaging => ({
  query,
  cursor: Option.none(),
  shown: NO_RESULTS,
  target: SEARCH_BATCH_SIZE,
  emptyPages: 0,
  shownBeforePage: 0,
})

/**
 * The settled query's Hugging Face results, evaluated in download order and revealed in a locked append-only order so
 * rows never reorder, jump, or disappear mid-list. A short head-of-line grace window prevents one slow repository from
 * blocking ready results behind blank skeletons. Reaching the end of a batch checks the next batch and fetches the
 * next Hugging Face page only when the current page runs out.
 */
function useSearchResults(
  settled: string,
  query: string,
  first: ExploreFirst | null,
  memory: ServingMemory,
  opened: Atom.Writable<string | null>,
  hovered: Atom.Writable<string | null>,
): ExploreResults {
  const client = useAgentClient()
  const registry = useContext(RegistryContext)
  const pinnedRepository = first?._tag === "Repository" ? first.repository : null
  const paging = useMemo(() => Atom.make<SearchPaging>(initialPaging("")), [])
  const setPaging = useAtomSet(paging)
  const releasedFrontier = useMemo(() => Atom.make<string | null>(null), [])
  const setReleasedFrontier = useAtomSet(releasedFrontier)
  const setOpened = useAtomSet(opened)
  const setHovered = useAtomSet(hovered)

  useEffect(() => {
    setPaging(initialPaging(settled))
    setReleasedFrontier(null)
    setOpened(null)
    setHovered(null)
  }, [settled, setPaging, setReleasedFrontier, setOpened, setHovered])

  const search = useMemo(() => {
    const currentAtom = Atom.make((get) => {
      const state = get(paging)
      return state.query === settled ? state : initialPaging(settled)
    })
    const pageAtom = Atom.make((get) => {
      const current = get(currentAtom)
      return settled === ""
        ? Result.initial<HuggingFaceSearchResponse>()
        : get(client.Models.SearchHuggingFaceModels({ query: settled, limit: Option.none(), cursor: current.cursor })).result
    })
    const pinnedAtom = Atom.make((get) => pinnedRepository === null || settled === ""
      ? Result.initial<HuggingFaceSearchResponse>()
      : get(client.Models.SearchHuggingFaceModels({ query: pinnedRepository, limit: Option.some(5), cursor: Option.none() })).result)
    const listedAtom = Atom.make((get) => {
      const current = get(currentAtom)
      const page = get(pageAtom)
      const pinnedPage = get(pinnedAtom)
      const pinnedWaiting = pinnedRepository !== null && (Result.isInitial(pinnedPage) || pinnedPage.waiting)
      if (pinnedWaiting && current.shown.length === 0) return NO_RESULTS
      const pageValue = Result.isSuccess(page) && !page.waiting ? page.value : null
      const pinned = Result.isSuccess(pinnedPage) ? pinnedPage.value.results.find(result => result.repository === pinnedRepository) : undefined
      return dedupeResults(listedFirst([...current.shown, ...(pageValue?.results ?? [])], pinned))
    })
    const targetAtom = Atom.make((get) => get(currentAtom).target)
    const scheduleAtom = searchSchedule(client, listedAtom, targetAtom, { releasedFrontier, pinnedRepository, opened, hovered })
    return Atom.make((get) => ({
      current: get(currentAtom),
      page: get(pageAtom),
      pinnedPage: get(pinnedAtom),
      listed: get(listedAtom),
      schedule: get(scheduleAtom),
    }))
  }, [client, settled, pinnedRepository, paging, releasedFrontier, opened, hovered])

  const { current, page, pinnedPage, listed, schedule } = useAtomValue(search)
  const pageLoading = Result.isInitial(page) || page.waiting
  const pinnedLoading = pinnedRepository !== null && (Result.isInitial(pinnedPage) || pinnedPage.waiting)
  const loading = pageLoading || (pinnedLoading && current.shown.length === 0)
  const pageValue = Result.isSuccess(page) && !page.waiting ? page.value : null
  const nextCursor = pageValue ? pageValue.nextCursor : Option.none<string>()
  const { frontier, checking, exhausted } = schedule.progress
  const revealedCount = schedule.revealed.length
  const frontierKey = frontier < listed.length ? listed[frontier]!.repository : null
  const frontierStarted = frontierKey !== null && schedule.checks.has(frontierKey)

  useEffect(() => {
    setReleasedFrontier(null)
    if (frontierKey === null || !checking || !frontierStarted) return
    const timer = setTimeout(() => setReleasedFrontier(frontierKey), FRONTIER_GRACE_MS)
    return () => clearTimeout(timer)
  }, [frontierKey, checking, frontierStarted, setReleasedFrontier])

  const consecutiveEmpty = revealedCount > current.shownBeforePage ? 0 : current.emptyPages + 1
  const canAutoAdvance = !loading && exhausted && Option.isSome(nextCursor) && consecutiveEmpty < MAX_AUTO_EMPTY_PAGES

  useEffect(() => {
    if (!canAutoAdvance || !pageValue || Option.isNone(nextCursor)) return
    setPaging((state) => {
      const base = state.query === settled ? state : initialPaging(settled)
      if (Option.isSome(base.cursor) && base.cursor.value === nextCursor.value) return state
      return {
        ...base,
        cursor: nextCursor,
        shown: dedupeResults([...base.shown, ...pageValue.results]),
        emptyPages: revealedCount > base.shownBeforePage ? 0 : base.emptyPages + 1,
        shownBeforePage: revealedCount,
      }
    })
  }, [canAutoAdvance, pageValue, nextCursor, settled, revealedCount, setPaging])

  const loadMore = () => {
    setPaging((state) => {
      const base = state.query === settled ? state : initialPaging(settled)
      const nextTarget = revealedCount + SEARCH_BATCH_SIZE
      if (frontier < listed.length || !pageValue || Option.isNone(nextCursor)) {
        return { ...base, target: nextTarget, emptyPages: 0 }
      }
      return {
        ...base,
        cursor: nextCursor,
        shown: dedupeResults([...base.shown, ...pageValue.results]),
        target: nextTarget,
        emptyPages: 0,
        shownBeforePage: revealedCount,
      }
    })
  }

  if (query !== settled || (loading && current.shown.length === 0)) return { _tag: "Searching" }
  if (Result.isFailure(page) && current.shown.length === 0) {
    return { _tag: "Failed", message: "Hugging Face search failed. Check your connection and try again." }
  }

  const byKey = new Map<string, HuggingFaceSearchResult>(listed.map(result => [result.repository, result]))
  const showUnchecked = (result: HuggingFaceSearchResult) =>
    result.gated || result.repository === pinnedRepository || listed.length === 1
  const rows: ExploreRow[] = [
    ...(first?._tag === "Offered" ? [{ _tag: "Offered" as const, model: first.model, prefetched: false }] : []),
    ...schedule.revealed.flatMap((key) => {
      const result = byKey.get(key)
      const check = schedule.checks.get(key)
      if (!result || !check) return []
      const retry = () => registry.refresh(client.Models.CheckHuggingFaceRepository(checkInput(result)))
      const row = searchRow(result, check, schedule.full.has(key), showUnchecked(result), retry, memory)
      return row ? [row] : []
    }),
  ]

  const hasMoreLocal = frontier < listed.length
  const hasMoreRemote = Option.isSome(nextCursor)
  const more: ExploreMore =
    loading || canAutoAdvance ? { _tag: "Loading" }
      : checking ? { _tag: "Checking" }
      : hasMoreLocal ? { _tag: "Auto", onMore: loadMore }
      : hasMoreRemote ? (exhausted ? { _tag: "Manual", onMore: loadMore } : { _tag: "Auto", onMore: loadMore })
      : null

  return { _tag: "Results", rows, more }
}

/** Explore models on the service's data: a Hugging Face search, and what to show before one. */
export function Explore(props: ExploreProps) {
  const settled = useDebounced(props.query.trim(), 350)
  const tooLong = new TextEncoder().encode(settled).length > HUGGING_FACE_SEARCH_QUERY_MAXIMUM_BYTES
  const opened = useMemo(() => Atom.make<string | null>(null), [])
  const hovered = useMemo(() => Atom.make<string | null>(null), [])
  const setOpened = useAtomSet(opened)
  const setHovered = useAtomSet(hovered)
  const searched = useSearchResults(tooLong ? "" : settled, props.query.trim(), props.first, props.memory, opened, hovered)
  const [selections, setSelections] = useState<Readonly<Record<string, DrafterSelection>>>({})
  const download = useDownload(selections)
  const drafters = drafterFor(selections, (key, selection) => setSelections(current => ({ ...current, [key]: selection })))
  const results: ExploreResults = props.query.trim() === ""
    ? { _tag: "Idle" }
    : tooLong ? { _tag: "Failed", message: "This search is longer than Hugging Face accepts. Shorten it." }
    : searched
  const useOpened = useMemo(() => useRepositoryQuants(props.memory), [props.memory])
  return <ExploreView {...props} results={results} download={download} drafters={drafters} useOpened={useOpened}
    onExpand={setOpened} onPrefetch={setHovered} />
}
