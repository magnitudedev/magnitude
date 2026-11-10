import { useEffect, useMemo, useState } from "react"
import { Option } from "effect"
import { Atom, Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { CheckIcon, CircleNotchIcon, DownloadSimpleIcon } from "@phosphor-icons/react"
import { HUGGING_FACE_SEARCH_QUERY_MAXIMUM_BYTES, repositoryDisplayName, type CatalogFormModelId, type HuggingFaceRepositoryId, type HuggingFaceSearchResult, type LocalModel } from "@magnitudedev/sdk"
import { formatStorageSize, useAgentClient, useLocalModelMutations, useLocalModels, type ServingMemory } from "@magnitudedev/client-common"
import { Button } from "../ui/button"
import { labForDeveloper } from "../model-logo"
import { onThisComputer, quantView, transferOf } from "./data"
import type { QuantView } from "./view"
import type { OfferedModel } from "./expanding-model"
import { ExploreModal, type ExploreResults, type RunnableFamily } from "./explore-modal"

function useDebounced<A>(value: A, delay: number): A {
  const [settled, setSettled] = useState(value)
  useEffect(() => {
    const timer = setTimeout(() => setSettled(value), delay)
    return () => clearTimeout(timer)
  }, [value, delay])
  return settled
}

const TRUSTED_PUBLISHERS = new Set(["unsloth", "bartowski", "ggml-org", "lmstudio-community"])

/** Whether a repository's publisher is one people rely on for quants, or the base model's own lab. */
const trustedPublisher = (repository: string, base: string | undefined) => {
  const owner = repository.split("/")[0]?.toLowerCase() ?? ""
  return TRUSTED_PUBLISHERS.has(owner) || (base !== undefined && base.split("/")[0]?.toLowerCase() === owner)
}

/** Search results grouped by the model they quantize: one offered model per base, one rung per repository's recommended quant. */
const offeredFromSearch = (results: readonly HuggingFaceSearchResult[], memory: ServingMemory): readonly OfferedModel[] => {
  const groups = new Map<string, HuggingFaceSearchResult[]>()
  for (const result of results) {
    // A result that doesn't run here, or that couldn't be checked, is no offer: search shows what runs.
    if (Option.isSome(result.unsupported) || Option.isNone(result.recommended)) continue
    const base = Option.getOrNull(result.base)
    const key = base && base.relation === "Quantized" ? base.repository : result.repository
    groups.set(key, [...groups.get(key) ?? [], result])
  }
  const models = [...groups.entries()].map(([key, members]): OfferedModel => {
    const lead = members[0]!
    const base = Option.getOrNull(lead.base)
    const developer = Option.getOrNull(lead.developer)
    // Publishers people rely on first, then the most downloaded: the order rows are listed in, and the first that
    // fits is the one offered.
    const ordered = [...members].sort((left, right) =>
      Number(trustedPublisher(right.repository, base?.repository)) - Number(trustedPublisher(left.repository, base?.repository))
        || right.downloads - left.downloads)
    const quants = ordered.flatMap(member => Option.match(member.recommended, {
      onNone: () => [],
      onSome: version => [quantView({ key: version.modelId, name: version.label, repository: member.repository, sizeBytes: version.sizeBytes, downloadBytes: version.downloadBytes }, version.assessment, memory)],
    }))
    const fastest = quants.find(quant => quant.fits)
    const first = members.flatMap(member => Option.toArray(member.recommended))[0]
    return {
      key,
      catalog: false,
      name: base && base.relation === "Quantized" ? repositoryDisplayName(base.repository) : lead.displayName,
      lab: developer ? labForDeveloper(developer.lab) : undefined,
      country: developer?.country ?? null,
      fineTuneOf: base && base.relation === "Finetune" ? repositoryDisplayName(base.repository) : null,
      drafter: null,
      contextLimit: first && first.assessment._tag === "Fits" ? first.assessment.profile.contextLength : null,
      quants,
      recommended: fastest?.key ?? quants[0]?.key ?? null,
    }
  })
  return models
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
function DownloadButton({ size, disabled, start }: { size: number; disabled: boolean; start: () => void }) {
  const [starting, setStarting] = useState(false)
  useEffect(() => {
    if (!starting) return
    const timer = setTimeout(() => setStarting(false), 30_000)
    return () => clearTimeout(timer)
  }, [starting])
  if (starting) return <Button disabled><CircleNotchIcon className="motion-safe:animate-spin" />Starting…</Button>
  return <Button disabled={disabled} onClick={() => { setStarting(true); start() }}><DownloadSimpleIcon />Download {formatStorageSize(size)}</Button>
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
    return <DownloadButton key={quant.key} size={quant.downloadBytes} disabled={!quant.fits} start={start} />
  }
}

/**
 * The settled query's Hugging Face results, a page at a time: pages already shown stay while the next one loads, and
 * reaching the end of the list asks for it. An empty query asks for nothing.
 */
function useSearchResults(settled: string, query: string, memory: ServingMemory): ExploreResults {
  const client = useAgentClient()
  const [pages, setPages] = useState<{ readonly query: string; readonly cursor: Option.Option<string>; readonly shown: readonly HuggingFaceSearchResult[] }>(
    { query: settled, cursor: Option.none(), shown: [] })
  const current = pages.query === settled ? pages : { query: settled, cursor: Option.none<string>(), shown: [] }
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
  return query !== settled || (loading && current.shown.length === 0) ? { _tag: "Searching" }
    : Result.isFailure(page) && current.shown.length === 0 ? { _tag: "Failed", message: "Hugging Face search failed. Check your connection and try again." }
    : {
        _tag: "Results",
        models: offeredFromSearch([...current.shown, ...(listed?.results ?? [])], memory),
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
  const searched = useSearchResults(tooLong ? "" : settled, props.query.trim(), props.memory)
  const download = useDownload()
  const results: ExploreResults = props.query.trim() === ""
    ? { _tag: "Recommended", models: props.recommended, machine: props.machine, measuring: props.measuring }
    : tooLong ? { _tag: "Failed", message: "This search is longer than Hugging Face accepts. Shorten it." }
    : searched
  return <ExploreModal {...props} results={results} download={download} />
}
