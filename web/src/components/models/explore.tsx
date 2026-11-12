import { useContext, useEffect, useMemo, useState } from "react"
import { Option } from "effect"
import { Atom, RegistryContext, Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { CheckIcon, CircleNotchIcon, DownloadSimpleIcon } from "@phosphor-icons/react"
import { HUGGING_FACE_SEARCH_QUERY_MAXIMUM_BYTES, repositoryDisplayName, type CatalogFormModelId, type HuggingFaceRepositoryCheck, type HuggingFaceRepositoryId, type HuggingFaceSearchResult, type LocalModel } from "@magnitudedev/sdk"
import { formatStorageSize, readSearchChecks, useAgentClient, useLocalModelMutations, useLocalModels, type QuickSearchCheck, type ServingMemory } from "@magnitudedev/client-common"
import { Button } from "../ui/button"
import { labForDeveloper } from "../model-logo"
import { downloadContents, onThisComputer, quantView, transferOf } from "./data"
import type { QuantView } from "./view"
import type { OfferedModel } from "./expanding-model"
import { ExploreModal, type ExploreResults, type ExploreRow, type RunnableFamily } from "./explore-modal"

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
  const quants = inspected.value.versions.map(version => quantView(
    { key: version.modelId, name: version.label, repository, sizeBytes: version.sizeBytes, downloadBytes: version.downloadBytes },
    Option.getOrNull(version.assessment), memory)).sort((left, right) => left.sizeBytes - right.sizeBytes)
  if (quants.length === 0) return model
  const recommended = inspected.value.versions.find(version => version.recommended)?.modelId ?? model.recommended
  return { ...model, quants, recommended, checking: false }
}

interface ExploreProps {
  readonly open: boolean
  readonly setOpen: (open: boolean) => void
  readonly query: string
  readonly setQuery: (value: string) => void
  readonly memory: ServingMemory
  readonly machine: string
  readonly recommended: readonly OfferedModel[]
  readonly measuring: { readonly settled: number; readonly total: number } | null
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

/** The footer of an opened model: its quant's Download, or how far that download is, or that it's here. */
function useDownload() {
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
    if (!model.catalog && quant.repository === null) return null
    const start = model.catalog
      ? () => install(quant.key as CatalogFormModelId)
      : () => add({
          repository: quant.repository as HuggingFaceRepositoryId,
          revision: Option.none(),
          version: Option.some(quant.name),
          drafter: { _tag: "Auto" },
          force: false,
        })
    return <DownloadButton key={quant.key} size={quant.downloadBytes} contents={downloadContents(model, quant)} disabled={!quant.fits} start={start} />
  }
}

/** No results yet: one list, so what is derived from a search's results doesn't change before they do. */
const NO_RESULTS: readonly HuggingFaceSearchResult[] = []

/**
 * The settled query's Hugging Face results, a page at a time: pages already shown stay while the next one loads, and
 * reaching the end of the list asks for it. An empty query asks for nothing. Its repositories are checked in the order
 * `readSearchChecks` runs them, `visible` being those on screen.
 */
function useSearchResults(settled: string, query: string, memory: ServingMemory, visible: Atom.Writable<ReadonlySet<string>>): ExploreResults {
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
  const listedResults = useMemo(() => [...current.shown, ...(listed?.results ?? [])], [current.shown, listed])
  const schedule = useAtomValue(useMemo(() => Atom.make((get) => {
    const byKey = new Map<string, HuggingFaceSearchResult>(listedResults.map(result => [result.repository, result]))
    const checks = new Map<string, Result.Result<HuggingFaceRepositoryCheck, unknown>>()
    const full = new Set<string>()
    readSearchChecks({
      listed: listedResults.map(result => result.repository),
      visible: get(visible),
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
    return { checks, full }
  }), [client, listedResults, visible]))
  const registry = useContext(RegistryContext)
  return query !== settled || (loading && current.shown.length === 0) ? { _tag: "Searching" }
    : Result.isFailure(page) && current.shown.length === 0 ? { _tag: "Failed", message: "Hugging Face search failed. Check your connection and try again." }
    : {
        _tag: "Results",
        rows: listedResults.flatMap((result) => {
          const retry = () => registry.refresh(client.Models.CheckHuggingFaceRepository(checkInput(result)))
          const row = searchRow(result, schedule.checks.get(result.repository) ?? Result.initial(), schedule.full.has(result.repository), retry, memory)
          return row ? [row] : []
        }),
        more: loading ? "Loading" : Option.isSome(next) ? more : null,
      }
}

/**
 * The Explore models modal on the service's data: Hugging Face search, and the recommendations it is given. One modal
 * whatever it shows, so typing a search changes its contents, never the modal itself.
 */
export function Explore(props: ExploreProps) {
  const settled = useDebounced(props.query.trim(), 350)
  const tooLong = new TextEncoder().encode(settled).length > HUGGING_FACE_SEARCH_QUERY_MAXIMUM_BYTES
  const visible = useMemo(() => Atom.make<ReadonlySet<string>>(new Set<string>()), [])
  const setVisible = useAtomSet(visible)
  const searched = useSearchResults(tooLong ? "" : settled, props.query.trim(), props.memory, visible)
  const download = useDownload()
  const results: ExploreResults = props.query.trim() === ""
    ? { _tag: "Recommended", models: props.recommended, machine: props.machine, measuring: props.measuring }
    : tooLong ? { _tag: "Failed", message: "This search is longer than Hugging Face accepts. Shorten it." }
    : searched
  const useOpened = useMemo(() => useRepositoryQuants(props.memory), [props.memory])
  return <ExploreModal {...props} results={results} download={download} useOpened={useOpened} onSeen={setVisible} />
}
