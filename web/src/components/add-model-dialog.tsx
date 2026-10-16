import { useId, useState, type ReactNode } from "react"
import { Cause, Exit, Option } from "effect"
import { Result, useAtomRefresh, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { ArrowLeftIcon, ArrowUpRightIcon, DownloadSimpleIcon, LinkIcon, MagnifyingGlassIcon, PlusIcon } from "@phosphor-icons/react"
import {
  candidateDownloadBytes, catalogVariantOfName, formatStorageSize, unsupportedReasonMessages, useAgentClient,
  type AddModelRequest,
} from "@magnitudedev/client-common"
import type { CandidateInspection, CandidateVersion, DrafterCandidate, HuggingFaceSearchResponse, HuggingFaceSearchResult, LocalModel, UnsupportedReason } from "@magnitudedev/sdk"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "./ui/dialog"
import { Button } from "./ui/button"
import { Input } from "./ui/input"
import { ErrorNotice, NoticeAction, type NoticeContent } from "./error-notice"
import { SkeletonLine } from "./page-skeletons"
import { HuggingFaceLogo, NamedLogo, variantLab } from "./model-logo"
import { DrafterList, drafterChoice, selectedDrafterCandidate, useDrafterCandidates, type DrafterSelection } from "./drafter-list"
import { candidateErrorNotice } from "./candidate-errors"
import { DirectoryBrowser, useDirectoryBrowser } from "./folder-picker"

/** What the dialog opens on: the app's own requests, plus Add GGUF file…, whose step starts with a file picker. */
export type AddDialogRequest = AddModelRequest | { readonly _tag: "LocalFile" }

type Step =
  | { readonly _tag: "Find"; readonly tab: FindTab; readonly query: string }
  | { readonly _tag: "Repository"; readonly reference: string; readonly file: Option.Option<string> }
  | { readonly _tag: "LocalFile" }

export type FindTab = "search" | "paste"

const initialStep = (request: AddDialogRequest): Step => {
  switch (request._tag) {
    case "Search": return { _tag: "Find", tab: "search", query: request.query }
    case "HuggingFace": return { _tag: "Repository", reference: request.repository, file: request.file }
    case "LocalFile": return { _tag: "LocalFile" }
  }
}

const huggingFaceUrl = /^https?:\/\/(www\.)?huggingface\.co\/[\w.-]+\/[\w.-]+(\/\S*)?$/
const repositoryReference = /^[\w.-]+\/[\w.-]+(:[\w.-]+)?$/

/** A pasted Hugging Face URL, `owner/repo`, or `owner/repo:QUANT`; anything else is refused before asking the service. */
export const parseModelReference = (text: string): Option.Option<string> => {
  const value = text.trim()
  return huggingFaceUrl.test(value) || repositoryReference.test(value) ? Option.some(value) : Option.none()
}

const downloadsFormat = new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 })
export const formatDownloads = (count: number) => `${downloadsFormat.format(count)} ${count === 1 ? "download" : "downloads"}`

/** The Add from Hugging Face dialog: Find (Search or Paste a link), then the repository step. */
export function AddModelDialog({ request, models, onClose, onAdded }: {
  request: Option.Option<AddDialogRequest>
  models: readonly LocalModel[]
  onClose: () => void
  onAdded: () => void
}) {
  return <Dialog open={Option.isSome(request)} onOpenChange={open => { if (!open) onClose() }}>
    <DialogContent className="max-h-[85vh] grid-cols-[minmax(0,1fr)] overflow-y-auto sm:max-w-xl">
      {Option.isSome(request) && <AddModelFlow key={JSON.stringify(request.value)} request={request.value} models={models} onAdded={onAdded} />}
    </DialogContent>
  </Dialog>
}

function AddModelFlow({ request, models, onAdded }: { request: AddDialogRequest; models: readonly LocalModel[]; onAdded: () => void }) {
  const [step, setStep] = useState(() => initialStep(request))
  const [find, setFind] = useState<{ tab: FindTab; query: string }>(() => step._tag === "Find" ? { tab: step.tab, query: step.query } : { tab: "search", query: "" })
  if (step._tag === "LocalFile") return <LocalFileStep models={models} onAdded={onAdded} />
  if (step._tag === "Find") return <FindStep tab={find.tab} query={find.query} models={models}
    onChange={next => setFind(next)} onOpen={reference => setStep({ _tag: "Repository", reference, file: Option.none() })} />
  return <RepositoryStep reference={step.reference} file={step.file} models={models} onAdded={onAdded}
    onBack={() => setStep({ _tag: "Find", tab: find.tab, query: find.query })} />
}

/** The search's observed state, as the find step renders it. */
export type SearchState =
  | { readonly _tag: "Idle" }
  | { readonly _tag: "Loading" }
  | { readonly _tag: "Failed"; readonly notice: NoticeContent }
  | { readonly _tag: "Loaded"; readonly response: HuggingFaceSearchResponse }

function FindStep({ tab, query, models, onChange, onOpen }: {
  tab: FindTab
  query: string
  models: readonly LocalModel[]
  onChange: (next: { tab: FindTab; query: string }) => void
  onOpen: (reference: string) => void
}) {
  const [typed, setTyped] = useState(query)
  const [pasted, setPasted] = useState("")
  const [pasteInvalid, setPasteInvalid] = useState(false)
  return <FindStepView tab={tab} onTab={next => onChange({ tab: next, query })}
    typed={typed} onType={setTyped} onSearch={() => onChange({ tab, query: typed.trim() })}
    results={query.trim() ? <SearchResults query={query.trim()} models={models} onOpen={onOpen} /> : null}
    pasted={pasted} pasteInvalid={pasteInvalid} onPaste={value => { setPasted(value); setPasteInvalid(false) }}
    onOpenPasted={() => Option.match(parseModelReference(pasted), { onNone: () => setPasteInvalid(true), onSome: onOpen })} />
}

function SearchResults({ query, models, onOpen }: { query: string; models: readonly LocalModel[]; onOpen: (reference: string) => void }) {
  const client = useAgentClient()
  const search = client.Models.SearchHuggingFaceModels({ query, limit: Option.none(), all: false })
  const result = useAtomValue(search).result
  const retry = useAtomRefresh(search)
  const state: SearchState = Result.isSuccess(result) ? { _tag: "Loaded", response: result.value }
    : Result.isFailure(result) ? { _tag: "Failed", notice: Option.match(Cause.failureOption(result.cause), { onNone: () => ({ title: "Couldn’t search Hugging Face", description: "Try again." }), onSome: candidateErrorNotice }) }
    : { _tag: "Loading" }
  return <SearchResultsView state={state} models={models} onOpen={onOpen} onRetry={retry} />
}

/** The find step: Search and Paste a link. */
export function FindStepView({ tab, onTab, typed, onType, onSearch, results, pasted, pasteInvalid, onPaste, onOpenPasted }: {
  tab: FindTab
  onTab: (tab: FindTab) => void
  typed: string
  onType: (value: string) => void
  onSearch: () => void
  results: ReactNode
  pasted: string
  pasteInvalid: boolean
  onPaste: (value: string) => void
  onOpenPasted: () => void
}) {
  const tabButton = (value: FindTab, label: string, icon: ReactNode) =>
    <Button role="tab" aria-selected={tab === value} variant={tab === value ? "secondary" : "ghost"} onClick={() => onTab(value)}>{icon}{label}</Button>
  return <>
    <DialogHeader>
      <DialogTitle className="flex items-center gap-2"><HuggingFaceLogo className="size-5" />Add a model from Hugging Face</DialogTitle>
      <DialogDescription>Any GGUF from a model family Magnitude supports: other quantizations, finetunes, and uncensored versions.</DialogDescription>
    </DialogHeader>
    <div role="tablist" aria-label="Find a model" className="flex items-center gap-1">
      {tabButton("search", "Search", <MagnifyingGlassIcon />)}
      {tabButton("paste", "Paste a link", <LinkIcon />)}
    </div>
    {tab === "search"
      ? <div role="tabpanel" aria-label="Search">
        <form onSubmit={event => { event.preventDefault(); onSearch() }}>
          <Input aria-label="Search Hugging Face" autoFocus placeholder="Search Hugging Face, for example qwen3.6 35b abliterated" value={typed} onChange={event => onType(event.target.value)} />
        </form>
        <div className="mt-3">{results ?? <p className="py-6 text-center text-sm text-slate-500">Search by model name, uploader, or variant. Press Enter to search.</p>}</div>
      </div>
      : <div role="tabpanel" aria-label="Paste a link">
        <form className="flex gap-2" onSubmit={event => { event.preventDefault(); onOpenPasted() }}>
          <Input aria-label="Hugging Face link" autoFocus aria-invalid={pasteInvalid || undefined} placeholder="https://huggingface.co/owner/repo" value={pasted} onChange={event => onPaste(event.target.value)} className="min-w-0 flex-1" />
          <Button type="submit" disabled={!pasted.trim()}>Open</Button>
        </form>
        <p className={`mt-2 text-xs ${pasteInvalid ? "text-red-600 dark:text-red-400" : "text-slate-500"}`}>{pasteInvalid ? "That isn’t a Hugging Face link. " : ""}Accepts a Hugging Face link, owner/repo, or owner/repo:QUANT.</p>
      </div>}
  </>
}

/** Supported repositories with uploader, lineage, and downloads, then the count hidden for unsupported architectures. */
export function SearchResultsView({ state, models, onOpen, onRetry }: { state: SearchState; models: readonly LocalModel[]; onOpen: (repository: string) => void; onRetry: () => void }) {
  switch (state._tag) {
    case "Idle": return null
    case "Loading": return <div aria-busy="true" aria-label="Searching Hugging Face" className="space-y-2 py-1">{[0, 1, 2].map(index => <SkeletonLine key={index} className="h-10" width={index % 2 ? "80%" : "95%"} />)}</div>
    case "Failed": return <ErrorNotice {...state.notice} actions={<NoticeAction onClick={onRetry}>Try again</NoticeAction>} />
    case "Loaded": {
      const { results, hiddenCount } = state.response
      return <div>
        {results.length === 0
          ? <p className="py-6 text-center text-sm text-slate-500">No supported GGUF repositories match this search.</p>
          : <ul aria-label="Search results" className="divide-y divide-slate-200 overflow-hidden rounded-lg border border-slate-200 dark:divide-slate-750 dark:border-slate-750">
            {results.map(result => <li key={result.repository}><SearchResultRow result={result} models={models} onOpen={() => onOpen(result.repository)} /></li>)}
          </ul>}
        {hiddenCount > 0 && <p className="mt-2 text-xs text-slate-500">{hiddenCount} more {hiddenCount === 1 ? "result uses an architecture" : "results use architectures"} Magnitude doesn’t run</p>}
      </div>
    }
  }
}

function SearchResultRow({ result, models, onOpen }: { result: HuggingFaceSearchResult; models: readonly LocalModel[]; onOpen: () => void }) {
  const variantOf = catalogVariantOfName(models, result.variantOf)
  return <button type="button" onClick={onOpen} className="flex w-full items-center gap-3 px-3 py-2.5 text-left hover:bg-slate-50 focus-visible:outline-2 focus-visible:outline-blue-500 dark:hover:bg-slate-800">
    <NamedLogo lab={variantLab(result.variantOf, result.displayName)} name={result.displayName} className="size-6" />
    <span className="min-w-0 flex-1">
      <span className="block truncate text-sm font-medium" title={result.repository}>{result.displayName}</span>
      <span className="block truncate text-xs text-slate-500">{result.uploader}{Option.isSome(variantOf) ? ` · Variant of ${variantOf.value}` : ""}</span>
    </span>
    <span className="shrink-0 whitespace-nowrap text-xs tabular-nums text-slate-500">{formatDownloads(result.downloads)}</span>
  </button>
}

function RepositoryStep({ reference, file, models, onBack, onAdded }: { reference: string; file: Option.Option<string>; models: readonly LocalModel[]; onBack?: () => void; onAdded: () => void }) {
  const client = useAgentClient()
  const inspect = client.Models.InspectModelCandidate({ reference })
  const inspection = useAtomValue(inspect).result
  const retry = useAtomRefresh(inspect)
  if (Result.isSuccess(inspection)) return <InspectedRepository inspection={inspection.value} file={file} models={models} onAdded={onAdded} {...(onBack ? { onBack } : {})} />
  return <>
    <DialogHeader><DialogTitle className="flex items-center gap-2">{onBack && <BackButton onBack={onBack} />}{reference}</DialogTitle></DialogHeader>
    {Result.isFailure(inspection)
      ? <InspectionFailure error={Cause.failureOption(inspection.cause)} onRetry={retry} />
      : <div aria-busy="true" aria-label="Reading the repository" className="space-y-2">
        <p className="text-xs text-slate-500">Reading the repository and checking each version against this machine…</p>
        {[0, 1, 2, 3].map(index => <SkeletonLine key={index} className="h-9" width="100%" />)}
      </div>}
  </>
}

function InspectionFailure({ error, onRetry }: { error: Option.Option<unknown>; onRetry: () => void }) {
  const notice = Option.match(error, { onNone: () => ({ title: "Couldn’t read this model", description: "Try again.", retryable: true }), onSome: candidateErrorNotice })
  return <ErrorNotice {...notice} actions={notice.retryable ? <NoticeAction onClick={onRetry}>Try again</NoticeAction> : undefined} />
}

function BackButton({ onBack }: { onBack: () => void }) {
  return <Button variant="ghost" size="icon-sm" aria-label="Back to search" onClick={onBack}><ArrowLeftIcon /></Button>
}

const plannedSupport = new Set<UnsupportedReason["_tag"]>([
  "WeightTypes", "RopeScaling", "UnrecognizedTensor", "UnrecognizedMetadata", "TiedOutputHead", "Tokenizer", "KernelDomain", "Backend",
])

/** The short verdict for versions the engine can't run: damaged, planned support, or not supported. */
export const unsupportedVersionLabel = (reasons: readonly UnsupportedReason[]): string =>
  reasons.some(reason => reason._tag === "Unreadable") ? "File is damaged"
    : reasons.length > 0 && reasons.every(reason => plannedSupport.has(reason._tag)) ? "Not supported yet"
    : "Not supported"

/** A version's engine verdict, never read from its file name; `detail` is the full reason behind a short label. */
export const versionVerdict = (version: CandidateVersion): { readonly label: string; readonly detail?: string; readonly selectable: boolean; readonly recommended: boolean } => {
  if (Option.isNone(version.assessment)) return { label: "Not checked yet", selectable: true, recommended: false }
  const assessment = version.assessment.value
  switch (assessment._tag) {
    case "Unsupported": return { label: unsupportedVersionLabel(assessment.reasons), detail: unsupportedReasonMessages(assessment.reasons).join(". "), selectable: false, recommended: false }
    case "DoesNotFit": return { label: "Too large for this machine", selectable: false, recommended: false }
    case "Fits": return version.recommended
      ? { label: "Recommended", selectable: true, recommended: true }
      : { label: "Fits your machine", selectable: true, recommended: false }
  }
}

/** The version shown first: a linked file, the pasted QUANT, the recommended one, then the first that can run. */
export const initialVersion = (inspection: CandidateInspection, file: Option.Option<string>): Option.Option<string> => {
  const linked = Option.flatMap(file, path => Option.fromNullable(inspection.versions.find(version => version.files.includes(path))))
  if (Option.isSome(linked)) return Option.some(linked.value.label)
  if (Option.isSome(inspection.selectedVersion)) return inspection.selectedVersion
  const preferred = inspection.versions.find(version => version.recommended) ?? inspection.versions.find(version => versionVerdict(version).selectable)
  return Option.fromNullable(preferred?.label)
}

/** Download names everything fetched: the version's target and projector bytes still to download, plus the chosen drafter. */
export const confirmAction = (inspection: CandidateInspection, version: CandidateVersion, drafter: Option.Option<DrafterCandidate>): { readonly label: string; readonly download: boolean } => {
  const { totalBytes, drafterBytes } = candidateDownloadBytes(version, inspection.projector, drafter)
  if (inspection.source._tag === "Local" || totalBytes === drafterBytes) {
    return { label: drafterBytes > 0 ? `Add (${formatStorageSize(drafterBytes)})` : "Add", download: false }
  }
  return { label: `Download (${formatStorageSize(totalBytes)})`, download: true }
}

function InspectedRepository({ inspection, file, models, onBack, onAdded }: { inspection: CandidateInspection; file: Option.Option<string>; models: readonly LocalModel[]; onBack?: () => void; onAdded: () => void }) {
  const [picked, setPicked] = useState(() => initialVersion(inspection, file))
  const version = Option.flatMap(picked, label => Option.fromNullable(inspection.versions.find(entry => entry.label === label)))
  const header = { inspection, models, selectedVersion: picked, onSelectVersion: (label: string) => setPicked(Option.some(label)), ...(onBack ? { onBack } : {}) }
  if (Option.isNone(version) || Option.isSome(inspection.unsupportedArchitecture)) return <RepositoryView {...header} />
  return <VersionWithDrafters key={version.value.label} version={version.value} header={header} onAdded={onAdded} />
}

function VersionWithDrafters({ version, header, onAdded }: {
  version: CandidateVersion
  header: Omit<Parameters<typeof RepositoryView>[0], "drafters" | "footer">
  onAdded: () => void
}) {
  const { drafters, footer } = useVersionConfirmation(header.inspection, version, onAdded)
  return <RepositoryView {...header} drafters={drafters} footer={footer} />
}

/** The drafter choices for one version and the footer that adds it with the chosen drafter. */
function useVersionConfirmation(inspection: CandidateInspection, version: CandidateVersion, onAdded: () => void) {
  const client = useAgentClient()
  const { candidates, otherPending, otherFailure, checkOther } = useDrafterCandidates({ _tag: "Candidate", source: inspection.source, files: version.files })
  const [selection, setSelection] = useState<DrafterSelection>({ _tag: "Auto" })
  const list = Result.isSuccess(candidates) ? candidates.value.candidates : []
  const findingDrafters = !Result.isSuccess(candidates) && !Result.isFailure(candidates)
  const add = useAtomSet(client.Models.AddHuggingFaceModel, { mode: "promiseExit" })
  const adding = useAtomValue(client.Models.AddHuggingFaceModel)
  const importFile = useAtomSet(client.Models.ImportModelFile, { mode: "promiseExit" })
  const importing = useAtomValue(client.Models.ImportModelFile)
  const source = inspection.source
  const state = source._tag === "Local" ? importing : adding
  const failure = Result.isFailure(state) && !state.waiting
    ? Option.match(Cause.failureOption(state.cause), { onNone: () => ({ title: "Couldn’t add this model", description: "Try again." }), onSome: candidateErrorNotice })
    : undefined
  const confirm = () => {
    const drafter = drafterChoice(selection, list)
    const done = (exit: Exit.Exit<unknown, unknown>) => { if (Exit.isSuccess(exit)) onAdded() }
    if (source._tag === "Local") void importFile({ path: source.path, drafter }).then(done)
    else void add({ repository: source.repository, revision: Option.some(source.commit), version: Option.some(version.label), drafter }).then(done)
  }
  return {
    drafters: Result.isFailure(candidates) ? <ErrorNotice severity="warning" title="Couldn’t list drafters" description="The recommended drafter is still used when you add this model." />
      : findingDrafters ? <div aria-busy="true" aria-label="Finding drafters" className="space-y-2"><p className="text-xs text-slate-500">Finding drafters for this version…</p><SkeletonLine className="h-10" width="100%" /><SkeletonLine className="h-10" width="100%" /></div>
      : <DrafterList candidates={list} selection={selection} onSelect={setSelection} otherPending={otherPending} otherFailure={otherFailure} onCheckOther={checkOther} />,
    footer: <RepositoryFooter local={source._tag === "Local"} action={confirmAction(inspection, version, selectedDrafterCandidate(selection, list))} pending={state.waiting || findingDrafters} failure={failure} onConfirm={confirm} />,
  }
}

const localFileNote = "Magnitude uses this file where it is and never moves or deletes it."

/** Add GGUF file…: the repository step with a file picker in place of the version list. */
function LocalFileStep({ models, onAdded }: { models: readonly LocalModel[]; onAdded: () => void }) {
  const browser = useDirectoryBrowser("gguf", Option.none())
  const [file, setFile] = useState(Option.none<string>())
  return <LocalFileView browser={<DirectoryBrowser browser={browser} selectedFile={file} onSelectFile={path => setFile(Option.some(path))} />}
    file={Option.map(file, path => <LocalFileInspection key={path} path={path} models={models} onAdded={onAdded} />)} />
}

/** The Add GGUF file step: the file picker on top, then the chosen file's verdict, drafters, and Add. */
export function LocalFileView({ browser, file }: { browser: ReactNode; file: Option.Option<ReactNode> }) {
  return <>
    <DialogHeader>
      <DialogTitle>Add a GGUF file</DialogTitle>
      <DialogDescription>A GGUF file on the computer running Magnitude. {localFileNote}</DialogDescription>
    </DialogHeader>
    <section aria-label="File" className="space-y-2">
      <h3 className="font-heading text-sm font-semibold">File</h3>
      {browser}
    </section>
    {Option.getOrElse(file, () => <p className="text-sm text-slate-500">Choose a GGUF file to see whether it runs on this machine.</p>)}
  </>
}

function LocalFileInspection({ path, models, onAdded }: { path: string; models: readonly LocalModel[]; onAdded: () => void }) {
  const client = useAgentClient()
  const inspect = client.Models.InspectModelCandidate({ reference: path })
  const inspection = useAtomValue(inspect).result
  const retry = useAtomRefresh(inspect)
  if (Result.isFailure(inspection)) return <InspectionFailure error={Cause.failureOption(inspection.cause)} onRetry={retry} />
  if (!Result.isSuccess(inspection)) return <div aria-busy="true" aria-label="Reading the file" className="space-y-2"><SkeletonLine className="h-9" width="100%" /><SkeletonLine className="h-10" width="100%" /></div>
  const version = inspection.value.versions[0]
  if (version === undefined || Option.isSome(inspection.value.unsupportedArchitecture) || !versionVerdict(version).selectable) {
    return <LocalFileSummary inspection={inspection.value} models={models} />
  }
  return <LocalFileVersion inspection={inspection.value} version={version} models={models} onAdded={onAdded} />
}

function LocalFileVersion({ inspection, version, models, onAdded }: { inspection: CandidateInspection; version: CandidateVersion; models: readonly LocalModel[]; onAdded: () => void }) {
  const { drafters, footer } = useVersionConfirmation(inspection, version, onAdded)
  return <LocalFileSummary inspection={inspection} models={models} drafters={drafters} footer={footer} />
}

/** The chosen file's name, lineage, size and one engine verdict, then its drafters and Add. */
export function LocalFileSummary({ inspection, models, drafters, footer }: { inspection: CandidateInspection; models: readonly LocalModel[]; drafters?: ReactNode; footer?: ReactNode }) {
  const variantOf = catalogVariantOfName(models, inspection.variantOf)
  const version = inspection.versions[0]
  const verdict = version === undefined ? undefined : versionVerdict(version)
  const facts = [...(Option.isSome(variantOf) ? [`Variant of ${variantOf.value}`] : []), ...Option.toArray(inspection.license)]
  return <>
    <section aria-label="Chosen file" className="flex items-center gap-3 rounded-lg border border-slate-200 px-3 py-2.5 dark:border-slate-750">
      <NamedLogo lab={variantLab(inspection.variantOf, inspection.displayName)} name={inspection.displayName} className="size-6" />
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm font-medium">{inspection.displayName}</span>
        {facts.length > 0 && <span className="block truncate text-xs text-slate-500">{facts.join(" · ")}</span>}
      </span>
      {version !== undefined && <span className="shrink-0 text-xs tabular-nums text-slate-500">{formatStorageSize(version.sizeBytes)}</span>}
      {verdict !== undefined && Option.isNone(inspection.unsupportedArchitecture) && (verdict.recommended
        ? <span className="shrink-0 rounded-full border border-blue-300 px-2 py-0.5 text-xs text-blue-700 dark:border-blue-700 dark:text-blue-400">Recommended</span>
        : <span className="shrink-0 text-xs text-slate-500" title={verdict.detail}>{verdict.label}</span>)}
    </section>
    {Option.isSome(inspection.unsupportedArchitecture)
      ? <ErrorNotice severity="info" title="Magnitude can’t run this model" description={`${unsupportedReasonMessages(inspection.unsupportedArchitecture.value).join(". ")}.`} />
      : version === undefined && <p className="text-sm text-slate-500">This file isn’t a GGUF model Magnitude can read.</p>}
    {drafters && <DrafterSection>{drafters}</DrafterSection>}
    {footer}
  </>
}

function DrafterSection({ children }: { children: ReactNode }) {
  return <section aria-label="Drafter">
    <h3 className="font-heading text-sm font-semibold">Drafter</h3>
    <p className="mb-2 text-xs text-slate-500">A small model that drafts tokens for this one to verify, for faster responses.</p>
    {children}
  </section>
}

/** Pinned-version note, any failure to add, and Download with its total or Add. */
export function RepositoryFooter({ local, action, pending, failure, onConfirm }: { local: boolean; action: { label: string; download: boolean }; pending: boolean; failure: NoticeContent | undefined; onConfirm: () => void }) {
  return <div className="space-y-3 border-t border-slate-200 pt-4 dark:border-slate-750">
    {failure && <ErrorNotice {...failure} />}
    <DialogFooter className="items-center sm:justify-between">
      <p className="min-w-0 text-xs text-slate-500">{local ? localFileNote : "Pinned to the current version of this repository"}</p>
      <Button className="shrink-0" disabled={pending} onClick={onConfirm}>{action.download ? <DownloadSimpleIcon /> : <PlusIcon />}{action.label}</Button>
    </DialogFooter>
  </div>
}

/** The repository step: header, versions with verdicts, drafter choices, and the footer. */
export function RepositoryView({ inspection, models, selectedVersion, onSelectVersion, onBack, drafters, footer }: {
  inspection: CandidateInspection
  models: readonly LocalModel[]
  selectedVersion: Option.Option<string>
  onSelectVersion: (label: string) => void
  onBack?: () => void
  drafters?: ReactNode
  footer?: ReactNode
}) {
  const name = useId()
  const variantOf = catalogVariantOfName(models, inspection.variantOf)
  const source = inspection.source
  const facts = [...(Option.isSome(variantOf) ? [`Variant of ${variantOf.value}`] : []), ...Option.toArray(inspection.license)]
  return <>
    <DialogHeader>
      <div className="flex items-start gap-3 pr-8">
        {onBack && <BackButton onBack={onBack} />}
        <NamedLogo lab={variantLab(inspection.variantOf, inspection.displayName)} name={inspection.displayName} className="mt-0.5 size-7" />
        <div className="min-w-0">
          <DialogTitle className="truncate">{inspection.displayName}</DialogTitle>
          <DialogDescription className="flex flex-wrap items-center gap-x-1.5">
            {source._tag === "HuggingFace"
              ? <a href={`https://huggingface.co/${source.repository}`} target="_blank" rel="noreferrer" className="inline-flex min-w-0 items-center gap-0.5 hover:underline"><span className="truncate">{source.repository}</span><ArrowUpRightIcon aria-hidden="true" className="size-3 shrink-0" /></a>
              : <span className="truncate font-mono" title={source.path}>{source.path}</span>}
            {facts.map(fact => <span key={fact} className="contents"><span aria-hidden="true">·</span><span>{fact}</span></span>)}
          </DialogDescription>
        </div>
      </div>
    </DialogHeader>
    {Option.isSome(inspection.unsupportedArchitecture)
      ? <ErrorNotice severity="info" title="Magnitude can’t run this model" description={`${unsupportedReasonMessages(inspection.unsupportedArchitecture.value).join(". ")}. Nothing was downloaded.`} />
      : inspection.versions.length === 0
        ? <p className="py-4 text-sm text-slate-500">This repository has no GGUF files.</p>
        : <>
          <section aria-label="Version">
            <h3 className="mb-2 font-heading text-sm font-semibold">Version</h3>
            <div role="radiogroup" aria-label="Version" className="divide-y divide-slate-200 overflow-hidden rounded-lg border border-slate-200 dark:divide-slate-750 dark:border-slate-750">
              {inspection.versions.map(version => {
                const verdict = versionVerdict(version)
                return <label key={version.label} className={`flex items-center gap-3 px-3 py-2 text-sm ${verdict.selectable ? "cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800" : "opacity-60"}`}>
                  <input type="radio" name={name} className="accent-blue-600" checked={Option.contains(selectedVersion, version.label)} disabled={!verdict.selectable} onChange={() => onSelectVersion(version.label)} />
                  <span className="w-24 shrink-0 truncate font-medium sm:w-28" title={version.files.join("\n")}>{version.label}</span>
                  <span className="w-16 shrink-0 tabular-nums text-slate-500 sm:w-20">{formatStorageSize(version.sizeBytes)}</span>
                  <span className="flex min-w-0 flex-1 flex-wrap items-center justify-end gap-x-2 gap-y-0.5 text-right text-xs">
                    {Option.isSome(version.onThisComputer) && <span className="text-green-700 dark:text-green-400">Already on this computer</span>}
                    {verdict.recommended
                      ? <span className="rounded-full border border-blue-300 px-2 py-0.5 text-blue-700 dark:border-blue-700 dark:text-blue-400">Recommended</span>
                      : <span className="text-slate-500" title={verdict.detail}>{verdict.label}</span>}
                  </span>
                </label>
              })}
            </div>
          </section>
          {drafters && <DrafterSection>{drafters}</DrafterSection>}
        </>}
    {footer}
  </>
}
