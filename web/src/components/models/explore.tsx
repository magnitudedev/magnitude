import { useEffect, useState } from "react"
import { Option } from "effect"
import { Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { DownloadSimpleIcon } from "@phosphor-icons/react"
import type { CatalogFormModelId, HuggingFaceRepositoryId, HuggingFaceSearchResult } from "@magnitudedev/sdk"
import { formatStorageSize, useAgentClient, useLocalModelMutations } from "@magnitudedev/client-common"
import { Button } from "../ui/button"
import { labForDeveloper } from "../model-logo"
import { quantView } from "./data"
import type { QuantView } from "./view"
import type { OfferedModel } from "./expanding-model"
import { ExploreModal, type ExploreResults, type RunnableFamily } from "./explore-modal"
import type { UnrunnableFile } from "./your-models"

function useDebounced<A>(value: A, delay: number): A {
  const [settled, setSettled] = useState(value)
  useEffect(() => {
    const timer = setTimeout(() => setSettled(value), delay)
    return () => clearTimeout(timer)
  }, [value, delay])
  return settled
}

/** Search results grouped by the model they quantize: one offered model per base, one rung per repository's recommended quant. */
const offeredFromSearch = (results: readonly HuggingFaceSearchResult[], unified: boolean): { readonly models: readonly OfferedModel[]; readonly unrunnable: readonly UnrunnableFile[]; readonly unchecked: readonly UnrunnableFile[] } => {
  const groups = new Map<string, HuggingFaceSearchResult[]>()
  const unrunnable: UnrunnableFile[] = []
  const unchecked: UnrunnableFile[] = []
  for (const result of results) {
    if (Option.isSome(result.unsupported)) {
      const architecture = Option.getOrNull(result.architecture)
      unrunnable.push({ name: result.repository, reason: architecture ? `${architecture} architecture` : "Not supported" })
      continue
    }
    if (Option.isNone(result.recommended)) {
      unchecked.push({ name: result.repository, reason: "Its files took too long to read" })
      continue
    }
    const base = Option.getOrNull(result.base)
    const key = base && base.relation === "Quantized" ? base.repository : result.repository
    groups.set(key, [...groups.get(key) ?? [], result])
  }
  const models = [...groups.entries()].map(([key, members]): OfferedModel => {
    const lead = members[0]!
    const base = Option.getOrNull(lead.base)
    const developer = Option.getOrNull(lead.developer)
    const quants = members.flatMap(member => Option.match(member.recommended, {
      onNone: () => [],
      onSome: version => [quantView({ key: version.modelId, name: version.label, repository: member.repository, sizeBytes: version.sizeBytes, downloadBytes: version.downloadBytes }, version.assessment, unified)],
    })).sort((left, right) => left.sizeBytes - right.sizeBytes)
    const fastest = quants.filter(quant => quant.fits).sort((left, right) => (right.speed ?? 0) - (left.speed ?? 0))[0]
    const first = members.flatMap(member => Option.toArray(member.recommended))[0]
    return {
      key,
      catalog: false,
      name: base && base.relation === "Quantized" ? base.repository.split("/").pop() ?? lead.displayName : lead.displayName,
      lab: developer ? labForDeveloper(developer.lab) : undefined,
      country: developer?.country ?? null,
      fineTuneOf: base && base.relation === "Finetune" ? base.repository.split("/").pop() ?? null : null,
      drafter: null,
      contextLimit: first && first.assessment._tag === "Fits" ? first.assessment.profile.contextLength : null,
      quants,
      recommended: fastest?.key ?? quants[0]?.key ?? null,
    }
  })
  return { models, unrunnable, unchecked }
}

interface ExploreProps {
  readonly open: boolean
  readonly setOpen: (open: boolean) => void
  readonly query: string
  readonly setQuery: (value: string) => void
  readonly unified: boolean
  readonly machine: string
  readonly recommended: readonly OfferedModel[]
  readonly measuring: { readonly settled: number; readonly total: number } | null
  readonly families: readonly RunnableFamily[]
  readonly owned: ReadonlySet<string>
}

function useDownload() {
  const client = useAgentClient()
  const add = useAtomSet(client.Models.AddHuggingFaceModel)
  const { install } = useLocalModelMutations()
  return (model: OfferedModel, quant: QuantView) => model.catalog
    ? <Button disabled={!quant.fits} onClick={() => install(quant.key as CatalogFormModelId)}><DownloadSimpleIcon />Download {formatStorageSize(quant.downloadBytes)}</Button>
    : quant.repository === null ? null : <Button disabled={!quant.fits} onClick={() => add({
    repository: quant.repository as HuggingFaceRepositoryId,
    revision: Option.none(),
    version: Option.some(quant.name),
    drafter: { _tag: "Auto" },
    force: false,
  })}><DownloadSimpleIcon />Download {formatStorageSize(quant.downloadBytes)}</Button>
}

/**
 * Explore while searching: the settled query's Hugging Face results, a page at a time. Pages already shown stay while
 * the next one loads; reaching the end of the list asks for it.
 */
function Searched({ settled, props }: { settled: string; props: ExploreProps }) {
  const client = useAgentClient()
  const [cursor, setCursor] = useState<Option.Option<string>>(Option.none())
  const [shown, setShown] = useState<readonly HuggingFaceSearchResult[]>([])
  const page = useAtomValue(client.Models.SearchHuggingFaceModels({ query: settled, limit: Option.none(), cursor })).result
  const download = useDownload()
  const loading = Result.isInitial(page) || page.waiting
  const current = Result.isSuccess(page) && !page.waiting ? page.value : null
  const next = current ? current.nextCursor : Option.none<string>()
  const more = () => {
    if (!current || Option.isNone(next)) return
    setShown(previous => [...previous, ...current.results])
    setCursor(next)
  }
  const results: ExploreResults = props.query.trim() !== settled || (loading && shown.length === 0) ? { _tag: "Searching" }
    : Result.isFailure(page) && shown.length === 0 ? { _tag: "Failed", message: "Hugging Face search failed. Check your connection and try again." }
    : {
        _tag: "Results",
        ...offeredFromSearch([...shown, ...(current?.results ?? [])], props.unified),
        more: loading ? "Loading" : Option.isSome(next) ? more : null,
      }
  return <ExploreModal {...props} results={results} download={download} />
}

/** The Explore models modal on the service's data: Hugging Face search, and the recommendations it is given. */
export function Explore(props: ExploreProps) {
  const settled = useDebounced(props.query.trim(), 350)
  const download = useDownload()
  return settled === ""
    ? <ExploreModal {...props} results={{ _tag: "Recommended", models: props.recommended, machine: props.machine, measuring: props.measuring }} download={download} />
    : <Searched settled={settled} props={props} />
}
