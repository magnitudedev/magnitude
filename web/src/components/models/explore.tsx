import { useEffect, useState } from "react"
import { Option } from "effect"
import { Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { DownloadSimpleIcon } from "@phosphor-icons/react"
import type { HuggingFaceRepositoryId, HuggingFaceSearchResult } from "@magnitudedev/sdk"
import { formatStorageSize, useAgentClient } from "@magnitudedev/client-common"
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
const offeredFromSearch = (results: readonly HuggingFaceSearchResult[], unified: boolean): { readonly models: readonly OfferedModel[]; readonly unrunnable: readonly UnrunnableFile[] } => {
  const groups = new Map<string, HuggingFaceSearchResult[]>()
  const unrunnable: UnrunnableFile[] = []
  for (const result of results) {
    if (Option.isSome(result.unsupported) || Option.isNone(result.recommended)) {
      const architecture = Option.getOrNull(result.architecture)
      unrunnable.push({ name: result.repository, reason: architecture ? `${architecture} architecture` : "Not supported" })
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
  return { models, unrunnable }
}

interface ExploreProps {
  readonly open: boolean
  readonly setOpen: (open: boolean) => void
  readonly query: string
  readonly setQuery: (value: string) => void
  readonly unified: boolean
  readonly machine: string
  readonly recommended: readonly OfferedModel[]
  readonly families: readonly RunnableFamily[]
  readonly owned: ReadonlySet<string>
}

function useDownload() {
  const client = useAgentClient()
  const add = useAtomSet(client.Models.AddHuggingFaceModel)
  return (quant: QuantView) => quant.repository === null ? null : <Button disabled={!quant.fits} onClick={() => add({
    repository: quant.repository as HuggingFaceRepositoryId,
    revision: Option.none(),
    version: Option.some(quant.name),
    drafter: { _tag: "Auto" },
    force: false,
  })}><DownloadSimpleIcon />Download {formatStorageSize(quant.downloadBytes)}</Button>
}

/** Explore while searching: the settled query's Hugging Face results. */
function Searched({ settled, props }: { settled: string; props: ExploreProps }) {
  const client = useAgentClient()
  const search = useAtomValue(client.Models.SearchHuggingFaceModels({ query: settled, limit: Option.none() })).result
  const download = useDownload()
  const results: ExploreResults = props.query.trim() !== settled || Result.isInitial(search) || search.waiting ? { _tag: "Searching" }
    : Result.isFailure(search) ? { _tag: "Failed", message: "Hugging Face search failed. Check your connection and try again." }
    : { _tag: "Results", ...offeredFromSearch(search.value.results, props.unified) }
  return <ExploreModal {...props} results={results} download={(_, quant) => download(quant)} />
}

/** The Explore models modal on the service's data: Hugging Face search, and the recommendations it is given. */
export function Explore(props: ExploreProps) {
  const settled = useDebounced(props.query.trim(), 350)
  const download = useDownload()
  return settled === ""
    ? <ExploreModal {...props} results={{ _tag: "Recommended", models: props.recommended, machine: props.machine }} download={(_, quant) => download(quant)} />
    : <Searched settled={settled} props={props} />
}
